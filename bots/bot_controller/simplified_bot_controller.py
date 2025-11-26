import base64
import json
import logging
import os
import signal
import threading
import time
from datetime import timedelta

import gi
import redis
import requests
from django.conf import settings
from django.utils import timezone

from bots.automatic_leave_configuration import AutomaticLeaveConfiguration
from bots.bot_adapter import BotAdapter
from bots.bots_api_utils import BotCreationSource
from bots.external_callback_utils import get_zoom_tokens
from bots.meeting_url_utils import meeting_type_from_url
from bots.models import (
    Bot,
    BotEventManager,
    BotEventSubTypes,
    BotEventTypes,
    BotStates,
    Credentials,
    MeetingTypes,
)
from bots.zoom_oauth_connections_utils import get_zoom_tokens_via_zoom_oauth_app

gi.require_version("GLib", "2.0")
from gi.repository import GLib

logger = logging.getLogger(__name__)


class AudioForwarder:
    """
    Forwards audio chunks to external AI service via HTTP.
    Implements connection pooling and error handling for reliable delivery.
    """

    def __init__(self, service_url=None, bot_id=None, meeting_id=None):
        self.service_url = service_url
        self.bot_id = bot_id
        self.meeting_id = meeting_id
        self.chunk_count = 0
        self.total_bytes = 0
        self.failed_requests = 0

        # Create a requests session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Configure retry behavior
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=requests.adapters.Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 503, 504],
            ),
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if self.service_url:
            logger.info(f"AudioForwarder initialized with service_url: {self.service_url}")
        else:
            logger.warning("AudioForwarder initialized without service_url - audio will only be logged")

    def forward_audio_chunk(self, speaker_id, chunk_time, chunk_bytes, sample_rate):
        """
        Forward audio chunk to external AI service.

        Args:
            speaker_id: Unique identifier for the speaker
            chunk_time: Timestamp of the audio chunk
            chunk_bytes: Raw audio data (PCM format)
            sample_rate: Sample rate of the audio (e.g., 48000)
        """
        self.chunk_count += 1
        self.total_bytes += len(chunk_bytes)

        # Log periodic stats
        if self.chunk_count % 100 == 0:
            logger.info(
                f"Audio forwarded - Speaker: {speaker_id}, "
                f"Chunks: {self.chunk_count}, "
                f"Total bytes: {self.total_bytes}, "
                f"Sample rate: {sample_rate}, "
                f"Failed: {self.failed_requests}"
            )

        # If no service URL configured, just log
        if not self.service_url:
            return

        try:
            # Encode audio data to base64
            audio_data_b64 = base64.b64encode(chunk_bytes).decode()

            # Prepare payload
            payload = {
                "bot_id": self.bot_id,
                "meeting_id": self.meeting_id,
                "speaker_id": speaker_id,
                "timestamp": chunk_time.isoformat(),
                "audio_data": audio_data_b64,
                "sample_rate": sample_rate,
                "duration_ms": None,  # Could calculate from chunk size
            }

            # Send to AI service
            response = self.session.post(
                f"{self.service_url}/audio/ingest",
                json=payload,
                timeout=5.0,  # 5 second timeout
            )

            response.raise_for_status()

            # Log success on first chunk and occasionally
            if self.chunk_count == 1 or self.chunk_count % 500 == 0:
                logger.info(
                    f"Successfully forwarded audio chunk {self.chunk_count} "
                    f"to AI service: {response.json()}"
                )

        except requests.exceptions.Timeout:
            self.failed_requests += 1
            logger.warning(
                f"Timeout forwarding audio chunk {self.chunk_count} "
                f"(speaker: {speaker_id})"
            )
        except requests.exceptions.RequestException as e:
            self.failed_requests += 1
            logger.error(
                f"Error forwarding audio chunk {self.chunk_count} "
                f"(speaker: {speaker_id}): {e}"
            )
        except Exception as e:
            self.failed_requests += 1
            logger.error(
                f"Unexpected error forwarding audio chunk {self.chunk_count}: {e}",
                exc_info=True
            )

    def cleanup(self):
        """Cleanup resources when bot shuts down."""
        logger.info(
            f"AudioForwarder cleanup - Total chunks: {self.chunk_count}, "
            f"Total bytes: {self.total_bytes}, "
            f"Failed requests: {self.failed_requests}"
        )

        # Close the requests session
        if self.session:
            self.session.close()


class SimplifiedBotController:
    """
    Simplified bot controller that only handles:
    - Joining meetings
    - Capturing audio input
    - Forwarding audio to external service
    - Basic bot lifecycle (join, leave, cleanup)
    """

    def __init__(self, bot_id):
        self.bot_in_db = Bot.objects.get(id=bot_id)
        self.cleanup_called = False
        self.run_called = False

        # Redis setup for bot control
        self.redis_client = None
        self.pubsub = None
        self.pubsub_channel = f"bot_{self.bot_in_db.id}"

        # Automatic leave configuration
        self.automatic_leave_configuration = AutomaticLeaveConfiguration(
            **self.bot_in_db.automatic_leave_settings()
        )

        # Audio forwarding service
        audio_service_url = self.bot_in_db.settings.get("audio_service_url")
        meeting_id = self.bot_in_db.settings.get("meeting_id")  # Get meeting_id from settings if available
        self.audio_forwarder = AudioForwarder(
            service_url=audio_service_url,
            bot_id=self.bot_in_db.object_id,
            meeting_id=meeting_id,
        )

        # Bot adapter (will be initialized in run())
        self.adapter = None
        self.main_loop = None

    def get_meeting_type(self):
        """Determine meeting platform from URL."""
        meeting_type = meeting_type_from_url(self.bot_in_db.meeting_url)
        if meeting_type is None:
            raise Exception(
                f"Could not determine meeting type for meeting url {self.bot_in_db.meeting_url}"
            )
        return meeting_type

    def get_per_participant_audio_sample_rate(self):
        """Get sample rate based on meeting platform."""
        meeting_type = self.get_meeting_type()
        if meeting_type == MeetingTypes.ZOOM:
            if self.bot_in_db.use_zoom_web_adapter():
                return 48000
            else:
                return 32000
        elif meeting_type == MeetingTypes.GOOGLE_MEET:
            return 48000
        elif meeting_type == MeetingTypes.TEAMS:
            return 48000

    def on_audio_chunk_received(self, speaker_id, chunk_time, chunk_bytes):
        """
        Called when audio chunk is received from bot adapter.
        Forwards audio to external service.
        """
        sample_rate = self.get_per_participant_audio_sample_rate()
        self.audio_forwarder.forward_audio_chunk(
            speaker_id, chunk_time, chunk_bytes, sample_rate
        )

    def get_google_meet_bot_adapter(self):
        """Initialize Google Meet bot adapter."""
        from bots.google_meet_bot_adapter import GoogleMeetBotAdapter

        return GoogleMeetBotAdapter(
            display_name=self.bot_in_db.name,
            send_message_callback=self.on_message_from_adapter,
            add_audio_chunk_callback=self.on_audio_chunk_received,
            meeting_url=self.bot_in_db.meeting_url,
            voice_agent_url=self.bot_in_db.voice_agent_url(),
            webpage_streamer_service_hostname=self.bot_in_db.k8s_webpage_streamer_service_hostname(),
            add_video_frame_callback=None,
            wants_any_video_frames_callback=None,
            add_mixed_audio_chunk_callback=None,
            upsert_caption_callback=None,
            upsert_chat_message_callback=None,
            add_participant_event_callback=None,
            automatic_leave_configuration=self.automatic_leave_configuration,
            add_encoded_mp4_chunk_callback=None,
            recording_view=self.bot_in_db.recording_view(),
            google_meet_closed_captions_language=None,
            should_create_debug_recording=False,
            start_recording_screen_callback=None,
            stop_recording_screen_callback=None,
            video_frame_size=self.bot_in_db.recording_dimensions(),
            record_chat_messages_when_paused=False,
            disable_incoming_video=True,
            google_meet_bot_login_is_available=False,
            google_meet_bot_login_should_be_used=False,
            create_google_meet_bot_login_session_callback=lambda: None,
        )

    def get_teams_bot_adapter(self):
        """Initialize Teams bot adapter."""
        from bots.teams_bot_adapter import TeamsBotAdapter

        return TeamsBotAdapter(
            display_name=self.bot_in_db.name,
            send_message_callback=self.on_message_from_adapter,
            add_audio_chunk_callback=self.on_audio_chunk_received,
            meeting_url=self.bot_in_db.meeting_url,
            voice_agent_url=self.bot_in_db.voice_agent_url(),
            webpage_streamer_service_hostname=self.bot_in_db.k8s_webpage_streamer_service_hostname(),
            add_video_frame_callback=None,
            wants_any_video_frames_callback=None,
            add_mixed_audio_chunk_callback=None,
            upsert_caption_callback=None,
            upsert_chat_message_callback=None,
            add_participant_event_callback=None,
            automatic_leave_configuration=self.automatic_leave_configuration,
            add_encoded_mp4_chunk_callback=None,
            recording_view=self.bot_in_db.recording_view(),
            teams_closed_captions_language=None,
            should_create_debug_recording=False,
            start_recording_screen_callback=None,
            stop_recording_screen_callback=None,
            video_frame_size=self.bot_in_db.recording_dimensions(),
            teams_bot_login_credentials=None,
            record_chat_messages_when_paused=False,
            disable_incoming_video=True,
        )

    def get_zoom_oauth_credentials_via_credentials_record(self):
        """Get Zoom OAuth credentials from database."""
        zoom_oauth_credentials_record = self.bot_in_db.project.credentials.filter(
            credential_type=Credentials.CredentialTypes.ZOOM_OAUTH
        ).first()
        if not zoom_oauth_credentials_record:
            raise Exception("Zoom OAuth credentials not found")

        zoom_oauth_credentials = zoom_oauth_credentials_record.get_credentials()
        if not zoom_oauth_credentials:
            raise Exception("Zoom OAuth credentials data not found")

        return zoom_oauth_credentials

    def get_zoom_oauth_credentials_via_zoom_oauth_app(self):
        """Get Zoom OAuth credentials from Zoom OAuth app."""
        zoom_oauth_app = self.bot_in_db.project.zoom_oauth_apps.first()
        if not zoom_oauth_app:
            return

        return {
            "client_id": zoom_oauth_app.client_id,
            "client_secret": zoom_oauth_app.client_secret,
        }

    def get_zoom_oauth_credentials_and_tokens(self):
        """Get Zoom OAuth credentials and tokens."""
        zoom_oauth_credentials = (
            self.get_zoom_oauth_credentials_via_zoom_oauth_app()
            or self.get_zoom_oauth_credentials_via_credentials_record()
        )

        zoom_tokens = {}
        if self.bot_in_db.zoom_tokens_callback_url():
            zoom_tokens = get_zoom_tokens(self.bot_in_db)
        else:
            zoom_tokens = get_zoom_tokens_via_zoom_oauth_app(self.bot_in_db)

        return zoom_oauth_credentials, zoom_tokens

    def get_zoom_web_bot_adapter(self):
        """Initialize Zoom Web bot adapter."""
        from bots.zoom_web_bot_adapter import ZoomWebBotAdapter

        zoom_oauth_credentials, zoom_tokens = (
            self.get_zoom_oauth_credentials_and_tokens()
        )

        return ZoomWebBotAdapter(
            display_name=self.bot_in_db.name,
            send_message_callback=self.on_message_from_adapter,
            add_audio_chunk_callback=self.on_audio_chunk_received,
            meeting_url=self.bot_in_db.meeting_url,
            voice_agent_url=self.bot_in_db.voice_agent_url(),
            webpage_streamer_service_hostname=self.bot_in_db.k8s_webpage_streamer_service_hostname(),
            add_video_frame_callback=None,
            wants_any_video_frames_callback=None,
            add_mixed_audio_chunk_callback=None,
            upsert_caption_callback=None,
            upsert_chat_message_callback=None,
            add_participant_event_callback=None,
            automatic_leave_configuration=self.automatic_leave_configuration,
            add_encoded_mp4_chunk_callback=None,
            recording_view=self.bot_in_db.recording_view(),
            should_create_debug_recording=False,
            start_recording_screen_callback=None,
            stop_recording_screen_callback=None,
            video_frame_size=self.bot_in_db.recording_dimensions(),
            zoom_client_id=zoom_oauth_credentials["client_id"],
            zoom_client_secret=zoom_oauth_credentials["client_secret"],
            zoom_closed_captions_language=None,
            should_ask_for_recording_permission=False,
            record_chat_messages_when_paused=False,
            disable_incoming_video=True,
            zoom_tokens=zoom_tokens,
        )

    def get_zoom_bot_adapter(self):
        """Initialize Zoom SDK bot adapter."""
        from bots.zoom_bot_adapter import ZoomBotAdapter

        zoom_oauth_credentials, zoom_tokens = (
            self.get_zoom_oauth_credentials_and_tokens()
        )

        return ZoomBotAdapter(
            use_one_way_audio=True,  # Only receive audio, don't send
            use_mixed_audio=False,
            use_video=False,
            display_name=self.bot_in_db.name,
            send_message_callback=self.on_message_from_adapter,
            add_audio_chunk_callback=self.on_audio_chunk_received,
            zoom_client_id=zoom_oauth_credentials["client_id"],
            zoom_client_secret=zoom_oauth_credentials["client_secret"],
            meeting_url=self.bot_in_db.meeting_url,
            add_video_frame_callback=None,
            wants_any_video_frames_callback=lambda: False,
            add_mixed_audio_chunk_callback=None,
            upsert_chat_message_callback=None,
            add_participant_event_callback=None,
            automatic_leave_configuration=self.automatic_leave_configuration,
            video_frame_size=self.bot_in_db.recording_dimensions(),
            zoom_tokens=zoom_tokens,
            zoom_meeting_settings=self.bot_in_db.zoom_meeting_settings(),
            record_chat_messages_when_paused=False,
        )

    def get_bot_adapter(self):
        """Get appropriate bot adapter based on meeting type."""
        meeting_type = self.get_meeting_type()
        if meeting_type == MeetingTypes.ZOOM:
            if self.bot_in_db.use_zoom_web_adapter():
                return self.get_zoom_web_bot_adapter()
            else:
                return self.get_zoom_bot_adapter()
        elif meeting_type == MeetingTypes.GOOGLE_MEET:
            return self.get_google_meet_bot_adapter()
        elif meeting_type == MeetingTypes.TEAMS:
            return self.get_teams_bot_adapter()

    def connect_to_redis(self):
        """Establish Redis connection for bot control."""
        if self.pubsub:
            self.pubsub.close()
        if self.redis_client:
            self.redis_client.close()

        redis_url = os.getenv("REDIS_URL") + (
            "?ssl_cert_reqs=none" if os.getenv("DISABLE_REDIS_SSL") else ""
        )
        self.redis_client = redis.from_url(redis_url)
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe(self.pubsub_channel)
        logger.info(f"Redis connection established for bot {self.bot_in_db.id}")

    def cleanup(self):
        """Cleanup bot resources and shut down gracefully."""
        if self.cleanup_called:
            logger.info("Cleanup already called, exiting")
            return
        self.cleanup_called = True

        logger.info(f"Starting cleanup for bot {self.bot_in_db.object_id}")

        # Cleanup audio forwarder
        if self.audio_forwarder:
            logger.info("Cleaning up audio forwarder...")
            self.audio_forwarder.cleanup()

        # Leave meeting and cleanup adapter
        if self.adapter:
            logger.info("Telling adapter to leave meeting...")
            self.adapter.leave()
            logger.info("Telling adapter to cleanup...")
            self.adapter.cleanup()

        # Quit main loop
        if self.main_loop and self.main_loop.is_running():
            self.main_loop.quit()

        logger.info(f"Cleanup completed for bot {self.bot_in_db.object_id}")

    def run(self):
        """Main entry point - start the bot."""
        if self.run_called:
            raise Exception("Run already called, exiting")
        self.run_called = True

        logger.info(f"Starting simplified bot controller for bot {self.bot_in_db.object_id}")

        # Connect to Redis for control commands
        self.connect_to_redis()

        # Initialize bot adapter
        self.adapter = self.get_bot_adapter()

        # Create GLib main loop
        self.main_loop = GLib.MainLoop()

        def repeatedly_try_to_reconnect_to_redis():
            """Retry Redis connection on failure."""
            reconnect_delay_seconds = 1
            num_attempts = 0
            while True:
                try:
                    self.connect_to_redis()
                    break
                except Exception as e:
                    logger.info(
                        f"Error reconnecting to Redis: {e} Attempt {num_attempts} / 30."
                    )
                    time.sleep(reconnect_delay_seconds)
                    num_attempts += 1
                    if num_attempts > 30:
                        raise Exception(
                            "Failed to reconnect to Redis after 30 attempts"
                        )

        def redis_listener():
            """Listen for Redis control commands in background thread."""
            while True:
                try:
                    message = self.pubsub.get_message(timeout=1.0)
                    if message:
                        GLib.idle_add(lambda: self.handle_redis_message(message))
                except Exception as e:
                    if isinstance(e, redis.exceptions.ConnectionError) and (
                        "Connection closed by server." in str(e)
                    ):
                        logger.info(
                            "Redis connection closed by server. Attempting to reconnect..."
                        )
                        repeatedly_try_to_reconnect_to_redis()
                    else:
                        logger.info(f"Error in Redis listener: {type(e)} {e}")
                        break

        # Start Redis listener thread
        redis_thread = threading.Thread(target=redis_listener, daemon=True)
        redis_thread.start()

        # Add timeout for bot lifecycle checks
        self.first_timeout_call = True
        GLib.timeout_add(100, self.on_main_loop_timeout)

        # Add signal handlers for graceful shutdown
        GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGTERM, self.handle_glib_shutdown)
        GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGINT, self.handle_glib_shutdown)

        # Run the main loop
        try:
            self.main_loop.run()
        except Exception as e:
            logger.error(f"Error in bot {self.bot_in_db.id}: {str(e)}")
            self.cleanup()
        finally:
            # Clean up Redis subscription
            if self.pubsub:
                self.pubsub.unsubscribe(self.pubsub_channel)
                self.pubsub.close()

    def take_action_based_on_bot_in_db(self):
        """Take action based on bot state (join, leave, etc.)."""
        if self.bot_in_db.state == BotStates.JOINING:
            logger.info("take_action_based_on_bot_in_db - JOINING")
            BotEventManager.set_requested_bot_action_taken_at(self.bot_in_db)
            self.adapter.init()
        elif self.bot_in_db.state == BotStates.LEAVING:
            logger.info("take_action_based_on_bot_in_db - LEAVING")
            BotEventManager.set_requested_bot_action_taken_at(self.bot_in_db)
            self.adapter.leave()
        elif self.bot_in_db.state == BotStates.STAGED:
            logger.info(
                f"take_action_based_on_bot_in_db - STAGED. join_at = {self.bot_in_db.join_at.isoformat()}"
            )

    def join_if_staged_and_time_to_join(self):
        """Join meeting if bot is staged and it's time to join."""
        if self.bot_in_db.state != BotStates.STAGED:
            return
        if self.bot_in_db.join_at > timezone.now() + timedelta(
            seconds=self.adapter.get_staged_bot_join_delay_seconds()
        ):
            return

        logger.info(
            f"Joining bot {self.bot_in_db.id} ({self.bot_in_db.object_id}) "
            f"because join_at is {self.bot_in_db.join_at.isoformat()}"
        )
        BotEventManager.create_event(
            bot=self.bot_in_db,
            event_type=BotEventTypes.JOIN_REQUESTED,
            event_metadata={"source": BotCreationSource.SCHEDULER},
        )
        self.take_action_based_on_bot_in_db()

    def set_bot_heartbeat(self):
        """Update bot heartbeat timestamp."""
        if (
            self.bot_in_db.last_heartbeat_timestamp is None
            or self.bot_in_db.last_heartbeat_timestamp
            <= int(timezone.now().timestamp()) - 60
        ):
            self.bot_in_db.set_heartbeat()

    def on_main_loop_timeout(self):
        """Periodic callback for bot lifecycle checks."""
        try:
            if self.first_timeout_call:
                logger.info("First timeout call - taking initial action")
                self.bot_in_db.refresh_from_db()
                self.take_action_based_on_bot_in_db()
                self.first_timeout_call = False

            # Set heartbeat
            self.set_bot_heartbeat()

            # Check if auto-leave conditions are met
            self.adapter.check_auto_leave_conditions()

            # For staged bots, check if it's time to join
            self.join_if_staged_and_time_to_join()

            return True

        except Exception as e:
            logger.error(f"Error in timeout callback: {e}")
            self.handle_exception_in_timeout_callback(e)
            return False

    def handle_exception_in_timeout_callback(self, e):
        """Handle exceptions in main loop timeout."""
        try:
            BotEventManager.create_event(
                bot=self.bot_in_db,
                event_type=BotEventTypes.FATAL_ERROR,
                event_sub_type=BotEventSubTypes.FATAL_ERROR_ATTENDEE_INTERNAL_ERROR,
                event_metadata={"error": str(e)},
            )
        except Exception as error:
            logger.error(f"Error in handle_exception_in_timeout_callback: {error}")
        self.cleanup()

    def handle_glib_shutdown(self):
        """Handle shutdown signals (SIGTERM, SIGINT)."""
        logger.info("handle_glib_shutdown called")

        try:
            BotEventManager.create_event(
                bot=self.bot_in_db,
                event_type=BotEventTypes.FATAL_ERROR,
                event_sub_type=BotEventSubTypes.FATAL_ERROR_PROCESS_TERMINATED,
            )
        except Exception as e:
            logger.error(f"Error creating FATAL_ERROR event: {e}")

        self.cleanup()
        return False

    def handle_redis_message(self, message):
        """Handle Redis control messages."""
        if message and message["type"] == "message":
            data = json.loads(message["data"].decode("utf-8"))
            command = data.get("command")

            if command == "sync":
                logger.info(f"Syncing bot {self.bot_in_db.object_id}")
                self.bot_in_db.refresh_from_db()
                self.take_action_based_on_bot_in_db()
            else:
                logger.info(f"Unknown command: {command}")

    def on_message_from_adapter(self, message):
        """Handle messages from bot adapter (meeting events)."""
        GLib.idle_add(lambda: self.take_action_based_on_message_from_adapter(message))

    def take_action_based_on_message_from_adapter(self, message):
        """Process messages from bot adapter."""
        message_type = message.get("message")

        # Meeting ended
        if message_type == BotAdapter.Messages.MEETING_ENDED:
            logger.info("Received message that meeting ended")
            if self.bot_in_db.state == BotStates.LEAVING:
                BotEventManager.create_event(
                    bot=self.bot_in_db, event_type=BotEventTypes.BOT_LEFT_MEETING
                )
            else:
                BotEventManager.create_event(
                    bot=self.bot_in_db, event_type=BotEventTypes.MEETING_ENDED
                )
            self.cleanup()
            return

        # Bot joined meeting
        if message_type == BotAdapter.Messages.BOT_JOINED_MEETING:
            logger.info("Received message that bot joined meeting")
            BotEventManager.create_event(
                bot=self.bot_in_db, event_type=BotEventTypes.BOT_JOINED_MEETING
            )
            return

        # Could not join meeting
        if message_type == BotAdapter.Messages.COULD_NOT_CONNECT_TO_MEETING:
            logger.info("Received message that could not connect to meeting")
            BotEventManager.create_event(
                bot=self.bot_in_db,
                event_type=BotEventTypes.COULD_NOT_JOIN,
                event_sub_type=BotEventSubTypes.COULD_NOT_JOIN_UNABLE_TO_CONNECT_TO_MEETING,
            )
            self.cleanup()
            return

        # Meeting not found
        if message_type == BotAdapter.Messages.MEETING_NOT_FOUND:
            logger.info("Received message that meeting not found")
            BotEventManager.create_event(
                bot=self.bot_in_db,
                event_type=BotEventTypes.COULD_NOT_JOIN,
                event_sub_type=BotEventSubTypes.COULD_NOT_JOIN_MEETING_MEETING_NOT_FOUND,
            )
            self.cleanup()
            return

        # Adapter requested leave
        if message_type == BotAdapter.Messages.ADAPTER_REQUESTED_BOT_LEAVE_MEETING:
            logger.info(
                f"Received message that adapter requested bot leave meeting reason={message.get('leave_reason')}"
            )
            event_sub_type_map = {
                BotAdapter.LEAVE_REASON.AUTO_LEAVE_SILENCE: BotEventSubTypes.LEAVE_REQUESTED_AUTO_LEAVE_SILENCE,
                BotAdapter.LEAVE_REASON.AUTO_LEAVE_ONLY_PARTICIPANT_IN_MEETING: BotEventSubTypes.LEAVE_REQUESTED_AUTO_LEAVE_ONLY_PARTICIPANT_IN_MEETING,
                BotAdapter.LEAVE_REASON.AUTO_LEAVE_MAX_UPTIME: BotEventSubTypes.LEAVE_REQUESTED_AUTO_LEAVE_MAX_UPTIME_EXCEEDED,
            }
            event_sub_type = event_sub_type_map.get(message.get("leave_reason"))
            BotEventManager.create_event(
                bot=self.bot_in_db,
                event_type=BotEventTypes.LEAVE_REQUESTED,
                event_sub_type=event_sub_type,
            )
            BotEventManager.set_requested_bot_action_taken_at(self.bot_in_db)
            self.adapter.leave()
            return

        # UI element not found
        if message_type == BotAdapter.Messages.UI_ELEMENT_NOT_FOUND:
            logger.info(f"Received message that UI element not found")
            BotEventManager.create_event(
                bot=self.bot_in_db,
                event_type=BotEventTypes.FATAL_ERROR,
                event_sub_type=BotEventSubTypes.FATAL_ERROR_UI_ELEMENT_NOT_FOUND,
                event_metadata={
                    "step": message.get("step"),
                    "current_time": message.get("current_time").isoformat() if message.get("current_time") else None,
                },
            )
            self.cleanup()
            return

        # Log unhandled messages
        logger.info(f"Received unhandled message from adapter: {message_type}")

