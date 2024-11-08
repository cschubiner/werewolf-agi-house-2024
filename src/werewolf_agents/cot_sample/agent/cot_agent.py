from typing import Any, Dict
from autogen import ConversableAgent, Agent, runtime_logging

import os,json,re
import asyncio
import logging
from collections import defaultdict

import openai
from openai import RateLimitError, OpenAI
from sentient_campaign.agents.v1.api import IReactiveAgent
from sentient_campaign.agents.v1.message import (
    ActivityMessage,
    ActivityResponse,
    TextContent,
    MimeType,
    ActivityMessageHeader,
    MessageChannelType,
)
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential,
)
GAME_CHANNEL = "play-arena"
WOLFS_CHANNEL = "wolf's-den"
MODERATOR_NAME = "moderator"
MODEL_NAME = "Llama31-70B-Instruct"

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger = logging.getLogger("demo_agent")
level = logging.DEBUG
logger.setLevel(level)
logger.propagate = True
handler = logging.StreamHandler()
handler.setLevel(level)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class CoTAgent(IReactiveAgent):
    # input -> thoughts -> init action -> reflection -> final action

    WOLF_PROMPT = """You are a wolf in a game of Werewolf. Your goal is to eliminate villagers without being detected. Consider the following:
    1. Blend in with villagers during day discussions.
    2. Coordinate with other werewolves to choose a target.
    3. Pay attention to the seer and doctor's potential actions.
    4. Defend yourself if accused, but don't be too aggressive.
    5. Never, under any circumstances, mention the wolf's den or any conversations that occur there."""

    VILLAGER_PROMPT = """You are a villager in a game of Werewolf. Your goal is to identify and eliminate the werewolves. Consider the following:
    1. Observe player behavior and voting patterns.
    2. Share your suspicions and listen to others.
    3. Be cautious of false accusations.
    4. Try to identify the seer and doctor to protect them."""

    SEER_PROMPT = """You are the seer in a game of Werewolf. Your ability is to learn one player's true identity each night. Consider the following:
    1. Use your knowledge wisely without revealing your role.
    2. Keep track of the information you gather each night.
    3. Guide village discussions subtly.
    4. Be prepared to reveal your role if it can save the village."""

    DOCTOR_PROMPT = """You are the doctor in a game of Werewolf. Your ability is to protect one player from elimination each night. Your goal is to always protect yourself, in all cases, no matter what. Consider the following:
    1. Always protect yourself every night.
    2. Do not protect others under any circumstances.
    3. Keep your role secret to avoid being targeted.
    4. Participate in discussions without revealing your role."""

    def __init__(self):
        logger.debug("WerewolfAgent initialized.")
        self.message_history = []  # Store messages as (header, content) tuples


    def __initialize__(self, name: str, description: str, config: dict = None):
        super().__initialize__(name, description, config)
        self._name = name
        self._description = description
        self.MODERATOR_NAME = MODERATOR_NAME
        self.WOLFS_CHANNEL = WOLFS_CHANNEL
        self.GAME_CHANNEL = GAME_CHANNEL
        self.config = config
        self.have_thoughts = True
        self.have_reflection = True
        self.role = None
        self.direct_messages = defaultdict(list)
        self.group_channel_messages = defaultdict(list)
        self.seer_checks = {}  # To store the seer's checks and results
        self.game_history = []  # To store the interwoven game history
        self.game_history_moderator = []  # To store moderator interactions
        self.werewolf_den_messages = []  # To store werewolf den chat messages
        self.message_count = 0  # Add counter for spam control
        self.fellow_werewolves = []  # Track fellow werewolves

        self.llm_config = self.sentient_llm_config["config_list"][0]
        self.openai_client = OpenAI(
            api_key=self.llm_config["api_key"],
            base_url=self.llm_config["llm_base_url"],
        )

        self.model = self.llm_config["llm_model_name"]
        logger.info(
            f"WerewolfAgent initialized with name: {name}, description: {description}, and config: {config}"
        )
        self.game_intro = None

    def _update_seer_checks(self, message_text):
        """
        Uses the LLM to parse the moderator's message to the Seer and updates the seer's checks.
        
        Args:
            message_text (str): The message content from the moderator.
        """
        prompt = f"""You are a Seer in a game of Werewolf. You received the following message from the moderator:

'{message_text}'

Extract the name of the player you investigated and their role (e.g., 'Villager', 'Werewolf').

Respond with the player's name and role in the following JSON format:

{{"player_name": "<player's name>", "role": "<player's role>"}}

Do not include any additional text. """

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an assistant that extracts information from game messages."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            json_response = response.choices[0].message.content.strip()
            logger.debug(f"LLM extraction: {json_response}")

            # Directly parse the JSON response
            data = json.loads(json_response)
            player = data["player_name"]
            role = data["role"]

            # Update self.seer_checks
            self.seer_checks[player] = role
            logger.info(f"Seer check updated via LLM: {player} is a {role}")

        except Exception as e:
            logger.error(f"Error using LLM to parse Seer's investigation result: {e}")

    async def async_notify(self, message: ActivityMessage):
        logger.info(f"ASYNC NOTIFY called with message: {message}")
        # Store message in history
        self.message_history.append((message.header, message.content.text))
        
        if message.header.channel_type == MessageChannelType.DIRECT:
            user_messages = self.direct_messages.get(message.header.sender, [])
            user_messages.append(message.content.text)
            self.direct_messages[message.header.sender] = user_messages
            self.game_history.append(f"[From - {message.header.sender}| To - {self._name} (me)| Direct Message]: {message.content.text}")
            
            if message.header.sender == self.MODERATOR_NAME:
                self.game_history_moderator.append(f"[From - {message.header.sender}| To - {self._name} (me)| Direct Message]: {message.content.text}")
                
                # If this is the first message, find the role
                if not len(user_messages) > 1:
                    self.role = self.find_my_role(message)
                    logger.info(f"Role found for user {self._name}: {self.role}")
                else:
                    # Handle Seer's investigation results
                    if self.role == 'seer':
                        self._update_seer_checks(message.content.text)
        else:
            group_messages = self.group_channel_messages.get(message.header.channel, [])
            group_messages.append((message.header.sender, message.content.text))
            self.group_channel_messages[message.header.channel] = group_messages
            self.game_history.append(f"[From - {message.header.sender}| To - Everyone| Group Message in {message.header.channel}]: {message.content.text}")
            if message.header.channel == self.WOLFS_CHANNEL:
                self.werewolf_den_messages.append(f"[From - {message.header.sender}| Group Message in {message.header.channel}]: {message.content.text}")
            if message.header.sender == self.MODERATOR_NAME:
                self.game_history_moderator.append(f"[From - {message.header.sender}| To - Everyone| Group Message in {message.header.channel}]: {message.content.text}")
            # if this is the first message in the game channel, the moderator is sending the rules, store them
            if message.header.channel == self.GAME_CHANNEL and message.header.sender == self.MODERATOR_NAME and not self.game_intro:
                self.game_intro = message.content.text
        logger.info(f"message stored in messages {message}")

    def get_interwoven_history_string(self, include_wolf_channel=False):
        interwoven_history_array = self.get_interwoven_history_array(include_wolf_channel)
        return "\n".join(interwoven_history_array)

    def get_interwoven_history_array(self, include_wolf_channel):
        """
        Retrieve the interwoven game history as a list of messages.
        
        Args:
            include_wolf_channel (bool): Whether to include messages from the wolf channel.
            
        Returns:
            List[str]: A list of game history messages.
        """
        return [event for event in self.game_history if
                include_wolf_channel or not event.startswith(f"[{self.WOLFS_CHANNEL}]")]

    def get_last_x_messages_from_interwoven_history_as_string(self, x: int, include_wolf_channel=False) -> str:
        """
        Retrieve the last x messages from the interwoven game history as a string.
        
        Args:
            x (int): The number of messages to retrieve.
            include_wolf_channel (bool): Whether to include messages from the wolf channel.
            
        Returns:
            str: A string containing the last x messages.
        """
        interwoven_history_array = self.get_interwoven_history_array(include_wolf_channel)
        last_x_messages = interwoven_history_array[-x:]
        return "\n".join(last_x_messages)

    @retry(
        wait=wait_exponential(multiplier=1, min=20, max=300),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(openai.RateLimitError),
    )
    def find_my_role(self, message):
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"The user is playing a game of werewolf as user {self._name}, help the user with question with less than a line answer",
                },
                {
                    "role": "user",
                    "name": self._name,
                    "content": f"You have got message from moderator here about my role in the werewolf game, here is the message -> '{message.content.text}', what is your role? possible roles are 'wolf','villager','doctor' and 'seer'. answer in a few words.",
                },
            ],
            temperature=0.0,
        )
        my_role_guess = response.choices[0].message.content
        logger.info(f"my_role_guess: {my_role_guess}")
        if "villager" in my_role_guess.lower():
            role = "villager"
        elif "seer" in my_role_guess.lower():
            role = "seer"
        elif "doctor" in my_role_guess.lower():
            role = "doctor"
        else:
            role = "wolf"
        
        return role

    async def async_respond(self, message: ActivityMessage):
        # This method is called when the agent needs to respond to a message that requires action (e.g., making a move or providing input).
        logger.info(f"ASYNC RESPOND called with message: {message}")
        
        # Store incoming message in history
        self.message_history.append((message.header, message.content.text))

        if message.header.channel_type == MessageChannelType.DIRECT and message.header.sender == self.MODERATOR_NAME:
            # Store the moderator's message in direct messages
            self.direct_messages[message.header.sender].append(message.content.text)
            # If the agent is the Seer, generate a response to investigate a player
            if self.role == "seer":
                response_message = self._get_response_for_seer_guess(message)
            # If the agent is the Doctor, generate a response to protect a player
            elif self.role == "doctor":
                response_message = self._get_response_for_doctors_save(message)
            
            # Create response header
            response_header = ActivityMessageHeader(
                message_id=f"response_to_{message.header.message_id}",
                sender=self._name,
                channel=message.header.channel,
                channel_type=message.header.channel_type,
                target_receivers=[message.header.sender]
            )
            # Store response in history
            self.message_history.append((response_header, response_message))
            
            response = ActivityResponse(response=response_message)
            # Log the conversation in game history 
            self.game_history.append(f"[From - {message.header.sender}| To - {self._name} (me)| Direct Message]: {message.content.text}")
            self.game_history.append(f"[From - {self._name} (me)| To - {message.header.sender}| Direct Message]: {response_message}")
            # Log moderator interactions
            if message.header.sender == self.MODERATOR_NAME:
                self.game_history_moderator.append(f"[From - {message.header.sender}| To - {self._name} (me)| Direct Message]: {message.content.text}")
                self.game_history_moderator.append(f"[From - {self._name} (me)| To - {message.header.sender}| Direct Message]: {response_message}")
        elif message.header.channel_type == MessageChannelType.GROUP:
            # Store the group message with sender information
            self.group_channel_messages[message.header.channel].append(
                (message.header.sender, message.content.text)
            )
            # If the message is in the main game channel
            if message.header.channel == self.GAME_CHANNEL:
                # Generate a discussion message or vote response for the common room
                response_message = self._get_discussion_message_or_vote_response_for_common_room(message)
            # If the message is in the wolf's den channel
            elif message.header.channel == self.WOLFS_CHANNEL:
                # Generate a response for the werewolves to decide on a target
                response_message = self._get_response_for_wolf_channel_to_kill_villagers(message)
            # Log the conversation in game history
            self.game_history.append(f"[From - {message.header.sender}| To - {self._name} (me)| Group Message in {message.header.channel}]: {message.content.text}")
            self.game_history.append(f"[From - {self._name} (me)| To - {message.header.sender}| Group Message in {message.header.channel}]: {response_message}")
            # Store werewolf den messages if applicable
            if message.header.channel == self.WOLFS_CHANNEL:
                self.werewolf_den_messages.append(f"[From - {message.header.sender}| Group Message in {message.header.channel}]: {message.content.text}")
                self.werewolf_den_messages.append(f"[From - {self._name} (me)| Group Message in {message.header.channel}]: {response_message}")
            # Log moderator interactions
            if message.header.sender == self.MODERATOR_NAME:
                self.game_history_moderator.append(f"[From - {message.header.sender}| To - {self._name} (me)| Group Message in {message.header.channel}]: {message.content.text}")
                self.game_history_moderator.append(f"[From - {self._name} (me)| To - {message.header.sender}| Group Message in {message.header.channel}]: {response_message}")
        
        # Return the ActivityResponse containing the agent's reply
        return ActivityResponse(response=response_message)


    
    def get_last_x_messages_from_werewolf_den_chat_as_string(self, x: int) -> str:
        """
        Retrieve the last `x` messages from the werewolf den chat as a string.
        
        Args:
            x (int): The number of messages to retrieve.
            
        Returns:
            str: A string containing the last `x` messages from the werewolf den chat.
        """
        last_messages = self.werewolf_den_messages[-x:]
        return "\n".join(last_messages)

    def get_last_x_messages_from_moderator_as_string(self, x: int) -> str:
        """
        Retrieve the last `x` messages exchanged with the moderator as a string.
        
        Args:
            x (int): The number of messages to retrieve.
            
        Returns:
            str: A string containing the last `x` messages with the moderator.
        """
        last_messages = self.game_history_moderator[-x:]
        return "\n".join(last_messages)

    def _summarize_game_history(self):
        self.detailed_history = "\n".join(self.game_history)

        # send the llm the previous summary of each of the other players and suspiciona nd information, the detailed chats of this day or night
        # llm will summarize the game history and provide a summary of the game so far
        # summarized game history is used for current situation

        pass


    def get_last_x_messages_from_seer_chat_as_string(self, x: int) -> str:
        """
        Retrieve the last 'x' messages from the Seer's direct messages with the moderator.
        """
        seer_chat = self.direct_messages.get(self.MODERATOR_NAME, [])
        last_messages = seer_chat[-x:]
        return "\n".join(last_messages)

    def get_messages_since_voting_began_as_string(self) -> str:
        """
        Retrieve all messages from message history since the most recent voting phase began.
        
        Returns:
            str: A string containing all messages since voting began.
        """
        messages_since_voting = []
        voting_start_found = False

        # Iterate backward through message history
        for header, content in reversed(self.message_history):
            # Check if message is from moderator and contains "vote"
            if header.sender == self.MODERATOR_NAME and "day vote" in content.lower():
                voting_start_found = True
                messages_since_voting.insert(0, (header, content))
                break
            else:
                messages_since_voting.insert(0, (header, content))
        
        if voting_start_found:
            # Format messages into strings
            formatted_messages = []
            for header, content in messages_since_voting:
                formatted_message = f"[From - {header.sender}| To - {', '.join(header.target_receivers) if header.target_receivers else 'Everyone'}| {header.channel_type.name} Message in {header.channel}]: {content}"
                formatted_messages.append(formatted_message)
            return "\n".join(formatted_messages)
        else:
            return "Voting phase has not begun yet."

    def _get_response_for_seer_guess(self, message):
        seer_checks_info = "\n".join([f"Checked {player}: {result}" for player, result in self.seer_checks.items()])
        seer_chat_history = self.get_last_x_messages_from_seer_chat_as_string(x=10)
        game_situation = f"My recent interactions with the moderator:\n{seer_chat_history}\n\nMy past seer checks:\n{seer_checks_info}"

        prompt = f"""{self.SEER_PROMPT}

{game_situation}

Based on your investigations and the information from the moderator, choose a player to investigate next. Choose someone you haven't checked before if possible.

Respond with the **name** of the player you choose to investigate, and no additional text."""

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are the Seer in a Werewolf game."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        action = response.choices[0].message.content.strip()
        logger.info(f"Seer action: {action}")
        return action

    def _get_response_for_doctors_save(self, message):
        # Doctor always protects themselves
        # action = self._name
        return f"I will protect myself ({self._name})."

    def _identify_fellow_werewolves_via_llm(self):
        """
        Use the LLM to identify fellow werewolves (allies) from the last seven messages
        in the werewolf den chat.
        """
        # Get the last 7 messages from the werewolf den
        last_messages = self.werewolf_den_messages[-7:]
        chat_history = "\n".join(last_messages)

        # Prepare the prompt
        prompt = f"""
You are analyzing the following chat history between allies in a secret group:

{chat_history}

From this conversation, list the names of your allies. Do not mention any roles or the word 'werewolf'. Respond with a comma-separated list of names only.
"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a player in a social deduction game."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            # Extract the names from the response
            allies_list = response.choices[0].message.content.strip()
            # Split the response into individual names and clean up
            allies = [name.strip() for name in allies_list.split(",")]
            # Update the fellow_werewolves list, excluding self
            self.fellow_werewolves = [name for name in allies if name != self._name]
            logger.info(f"Identified fellow allies: {self.fellow_werewolves}")
        except Exception as e:
            logger.error(f"Error identifying fellow allies: {e}")

    def _get_discussion_message_or_vote_response_for_common_room(self, message):
        # Check if this is a vote request based on last moderator message
        last_moderator_message = self.game_history_moderator[-1] if self.game_history_moderator else ""
        if "vote" in last_moderator_message.lower():
            return self._get_vote_response_for_common_room(message)
        else:
            return self._get_discussion_message_for_common_room(message)

    def _get_discussion_message_for_common_room(self, message):
        # Prepare the role-specific prompt
        role_prompt = getattr(self, f"{self.role.upper()}_PROMPT", self.VILLAGER_PROMPT)

        # Include additional logic for the seer role
        if self.role == "seer":
            identified_werewolves = [player for player, role in self.seer_checks.items() if role.lower() == 'werewolf']
            if identified_werewolves:
                accused_player = identified_werewolves[-1]
                role_prompt += f"""
Important:
- You have strong evidence that {accused_player} is a werewolf based on your observations.
- Accuse {accused_player} vigorously of being a werewolf.
- Mention that their behavior has been very suspicious, because of what they said in the past round AND the one before.
"""
            else:
                seer_checks_info = "\n".join([f"{player}: {role}" for player, role in self.seer_checks.items()])
                role_prompt += f"""
My past investigations:
{seer_checks_info}
"""

        # Add special instructions for werewolves
        if self.role == "wolf":
            if not self.fellow_werewolves:
                self._identify_fellow_werewolves_via_llm()
            fellow_werewolves_str = ', '.join(self.fellow_werewolves)
            role_prompt += f"""
Important:
- Never accuse your fellow allies: {fellow_werewolves_str}.
- Do not vote to eliminate them.
- If they are accused, shift blame to someone else without explicitly defending your ally.
- Focus on accusing other players.
- Keep the fact that you are an ally secret.
"""

        # Get the recent game situation
        game_situation = self.get_last_x_messages_from_moderator_as_string(x=2)

        # Construct the prompt for discussion
        prompt = f"""{role_prompt}

Current game situation: '''
{game_situation}'''

Based on the current game situation, participate in the discussion.

Respond accordingly."""

        # Call the LLM to generate a discussion message
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are the role of 'villager' in a Werewolf game. You are but a poor villager with not enough means to tie my own shoestring. Mention that in some of your responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        action = response.choices[0].message.content.strip()
        logger.info(f"ZZZZ WWWW - Discussion prompt: {prompt}")
        logger.info(f"Discussion action: {action}")
        return action

    def _get_vote_response_for_common_room(self, message):
        # Prepare the role-specific prompt
        role_prompt = getattr(self, f"{self.role.upper()}_PROMPT", self.VILLAGER_PROMPT)

        # Include additional logic for voting
        if self.role == "seer":
            identified_werewolves = [player for player, role in self.seer_checks.items() if role.lower() == 'werewolf']
            if identified_werewolves:
                accused_player = identified_werewolves[-1]
                role_prompt += f"""
Important:
- You have strong evidence that {accused_player} is a werewolf based on your observations.
- Vote to eliminate {accused_player} and encourage others to do the same.
"""
            else:
                role_prompt += """
Important:
- Vote based on your suspicions from the discussions.
"""

        if self.role == "wolf":
            if not self.fellow_werewolves:
                self._identify_fellow_werewolves_via_llm()
            fellow_werewolves_str = ', '.join(self.fellow_werewolves)
            role_prompt += f"""
Important:
- Never vote to eliminate this/these players: {fellow_werewolves_str}.
- Vote to eliminate other players. 
- If there are other votes in the chat history above, gang up on the same player (as long as they're not you ({self._name}) or your allies).
"""

        # Get the messages since voting began
        game_situation = self.get_messages_since_voting_began_as_string()

        # logger.error("zzzzzzzzzzzzzzzzzzzzzzzzzzzwwwwwwwwwwwwwwwwwwwwwwwwww1111111111111111111111")
        # logger.error(game_situation)
        # logger.error("zzzzzzzzzzzzzzzzzzzzzzzzzzzwwwwwwwwwwwwwwwwwwwwwwwwww2222222222222222222222")

        # Construct the prompt for voting
        prompt = f"""{role_prompt}

Current game situation: '''
{game_situation}'''

Based on the current game situation, decide on a player to vote for elimination.

Respond with the **name** of the player you choose to eliminate, and optionally include very brief reasoning."""

        # Call the LLM to generate a vote response
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a 'villager' role in a Werewolf game."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        action = response.choices[0].message.content.strip()
        logger.info(f"ZZZZZ YYYYY - Vote prompt: {prompt}")
        logger.info(f"Vote action: {action}")
        return action

    def _get_response_for_wolf_channel_to_kill_villagers(self, message):
        if self.role != "wolf":
            return "I am not a werewolf and cannot participate in this channel."

        # game_situation = self.get_last_x_messages_from_interwoven_history_as_string(x=2, include_wolf_channel=True)
        game_situation = self.get_last_x_messages_from_werewolf_den_chat_as_string(x=2)

        prompt = f"""{self.WOLF_PROMPT}

Current game situation:
{game_situation}

Based on the current game situation, suggest a target for elimination.

Note: Once again, if you're prompted to vote, respond with the **name** of the player you choose to eliminate, and optionally include very brief reasoning. You MUST do this.

Respond with the **name** of the player you suggest to eliminate, and optionally include your reasoning."""

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a werewolf in a Werewolf game."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        action = response.choices[0].message.content.strip()
        logger.info(f"Werewolf action: {action}")
        return action
