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
1. Vote to kill whoever you believe is the seer. If you do not have a good guess, vote for whoever you believe is most likely to be a villager. Vote for the suspected doctor last. NEVER vote for your fellow werewolf.
2. Note that other players can try to trick you. Just because someone says they are the seer, doesn't mean that they actually are."""

    VILLAGER_PROMPT = """You are a villager in a game of Werewolf. Your goal is to identify and eliminate the werewolves. Consider the following:
1. Observe quietly and take mental notes of behavior patterns.
2. Listen more than you speak.
3. Share observations without making direct accusations.
4. Support others' reasonable suspicions rather than leading accusations.

Important:
- During the voting phase, respond with only the name of the player you choose to eliminate, and no additional text."""

    SEER_PROMPT = """You are the seer in a game of Werewolf. Your ability is to learn one player's true identity each night. Consider the following:
1. Keep your role secret as long as possible.
2. Guide discussions indirectly using your knowledge.
3. Present suspicions as hunches rather than certainties.
4. Only reveal your role as an absolute last resort."""

    DOCTOR_PROMPT = """You are the doctor in a game of Werewolf. Your ability is to protect one player from elimination each night. Your goal is to help the village by drawing the werewolves' attacks onto yourself. Consider the following:
    1. Always protect yourself every night.
    2. Always reveal your role during discussions.
    3. Emphasize that you are healing yourself every night and cannot be killed at night.
    4. Encourage the werewolves to target you at night.
    5. Assure the villagers that you are an important ally and should not be eliminated."""

    def __init__(self):
        logger.debug("WerewolfAgent initialized.")
        self.message_history = []  # Store messages as (header, content) tuples

    def _generate_role_guesses(self, game_situation: str, alive_players: str) -> str:
        """
        Generates guesses of roles for all alive players, including self, using the LLM.
        Returns a neatly formatted string of the guesses.
        """
        # Prepare the seer checks information if role is 'seer'
        seer_checks_info = ""
        if self.role == 'seer':
            seer_checks_info = "\n".join([f"{player}: {role}" for player, role in self.seer_checks.items()])

        # Set the role to 'Villager' in the prompt to avoid being found out
        role_in_prompt = 'Villager'

        # Prepare the prompt for the LLM
        prompt = f"""
You are '{self._name}' in a game of Werewolf. Your role is '{role_in_prompt}'

Your task is to analyze the game situation and guess the roles of all alive players, including yourself.

Instructions:
- For yourself, you know your role with 100% certainty; state it with 100% confidence.
- For other players, provide your best guess of their roles and assign a percentage likelihood to each role.
- Ensure to include at least one werewolf and at least one seer in your guesses.
- Only include players who are currently alive.

Game Situation:
{game_situation}

List of Alive Players:
{alive_players}

Seer Checks Information:
{seer_checks_info}

Respond with your guesses in the following neatly formatted manner:

Player Name: Role Guess (Confidence%)
e.g.,
Alice: Villager (50%)
Bob: Werewolf (70%)

Do not include any additional text.
"""
        # Call the LLM to generate the role guesses
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are '{self._name}', a player in a game of Werewolf."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        role_guesses = response.choices[0].message.content.strip()
        logger.info(f"Role guesses:\n{role_guesses}")
        return role_guesses

    def _get_alive_players_via_llm(self) -> str:
        """
        Use the LLM to extract the names of alive players from the last six messages
        of the interwoven chat history.

        Returns:
            str: A comma-separated list of alive player names as provided by the LLM.
        """
        # Get the last six messages from the interwoven chat history
        last_six_messages = self.get_last_x_messages_from_interwoven_history_as_string(6)

        # Prepare the prompt
        prompt = f"""
You are analyzing a Werewolf game. Based on the following chat history, list all the names of players who are currently alive in the game.

Respond with a comma-separated list of names, and no additional text.

Chat History:
{last_six_messages}
"""

        # Call the LLM to get the alive players
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an assistant that extracts information from game messages."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            alive_players = response.choices[0].message.content.strip()
            return alive_players
        except Exception as e:
            logger.error(f"Error fetching alive players via LLM: {e}")
            return ""


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
                    "content": f"You are '{self._name}' in a game of Werewolf.",
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

    def get_messages_since_day_start_as_string(self, exclude_senders=None) -> str:
        """
        Retrieve all messages from message history since the most recent day start.
        
        Args:
            exclude_senders: Optional list of sender names whose messages should be excluded
            
        Returns:
            str: A string containing all messages since day start began.
        """
        messages_since_day_start = []
        day_start_found = False

        # Find the index where "day start" is announced
        for i, (header, content) in enumerate(self.message_history):
            if header.sender == self.MODERATOR_NAME and "day start" in content.lower():
                day_start_found = True
                day_start_index = i + 1  # Start collecting messages after this index
                break

        if not day_start_found:
            return "Day start not found in message history."

        # Collect messages after "day start"
        for header, content in self.message_history[day_start_index:]:
            if exclude_senders is None or header.sender not in exclude_senders:
                formatted_message = f"[From - {header.sender}| To - {', '.join(header.target_receivers) if header.target_receivers else 'Everyone'}| {header.channel_type.name} Message in {header.channel}]: {content}"
                messages_since_day_start.append(formatted_message)

        return "\n".join(messages_since_day_start)

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
                {
                    "role": "system",
                    "content": f"You are '{self._name}', the Seer in a Werewolf game."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        action = response.choices[0].message.content.strip()
        logger.info(f"Seer action: {action}")
        return action

    def _get_response_for_doctors_save(self, message):
        # Doctor always protects themselves to remain invincible
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

    def _detect_accusations_against_me(self) -> str:
        """
        Uses the LLM to detect if someone has accused us or is voting for us.
        Analyzes messages from today only.
        Returns the severity level: 'NONE', 'NOT_MENTIONED', 'MILD_ACCUSATION', 'HEAVY_ACCUSATION'.
        """
        try:
            # Get messages from today only
            today_messages = self.get_messages_since_day_start_as_string()

            # if my name is not in any of today_messages, return NOT_MENTIONED
            if self._name.lower() not in today_messages.lower():
                return "NOT_MENTIONED"

            # Prepare the prompt for the LLM
            prompt = f"""
You are analyzing a conversation between players in a Werewolf game.
Your task is to determine if any player has accused or is voting against you ('{self._name}') in the following messages:

<messages>
{today_messages}
</messages>


Classify the severity of accusations towards '{self._name}' into one of the following categories:
- NONE: No accusations or mentions towards the player.
- NOT_MENTIONED: The player's name is not mentioned at all.
- MILD_ACCUSATION: Slight or indirect accusations or suspicions towards the player.
- HEAVY_ACCUSATION: Direct and strong accusations or explicit votes against the player.

To analyze the messages:
1. Look for any mentions of the player's name in the conversation.
2. If the name is not mentioned at all, classify as NOT_MENTIONED.
3. If the name is mentioned, carefully read the context to determine if there are any accusations or suspicions.
4. Evaluate the strength and directness of any accusations or suspicions found.
5. Choose the most appropriate classification based on your analysis.

Provide your response in the following format:
<classification>INSERT CLASSIFICATION HERE</classification>

Your response should contain only the classification tag with one of the four categories (NONE, NOT_MENTIONED, MILD_ACCUSATION, or HEAVY_ACCUSATION) inside. Do not include any additional text or explanation.
"""

            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are {self._name}, a player in a game of werewolf."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM response: {response_text}")

            # Use regex to extract the classification between the tags
            match = re.search(r'<classification>(.*?)</classification>', response_text, re.IGNORECASE)
            if match:
                severity = match.group(1).strip().upper()
                # Check if the extracted severity is one of the expected values
                if severity not in ["NONE", "NOT_MENTIONED", "MILD_ACCUSATION", "HEAVY_ACCUSATION"]:
                    logger.warning(f"Unexpected classification value: '{severity}'. Defaulting to 'NONE'.")
                    severity = "NONE"
            else:
                logger.warning("Failed to parse classification from LLM response. Defaulting to 'NONE'.")
                severity = "NONE"

            logger.info(f"Detected accusation severity: {severity}")
        except Exception as e:
            logger.error(f"Error detecting accusations: {e}")
            severity = "NONE"

        return severity

    def _get_players_who_have_spoken_since_day_start(self):
        """
        Returns a list of player names who have sent messages since day start.
        """
        day_start_found = False
        players_who_spoke = set()
        for header, content in self.message_history:
            if header.sender == self.MODERATOR_NAME and "day start" in content.lower():
                day_start_found = True
                continue  # Start collecting messages after day start
            if day_start_found:
                # Consider messages from other players (excluding self and moderator)
                if header.sender != self.MODERATOR_NAME and header.sender != self._name:
                    players_who_spoke.add(header.sender)
        return list(players_who_spoke)

    def _get_discussion_message_or_vote_response_for_common_room(self, message):
        # Detect accusations against us
        accusation_severity = self._detect_accusations_against_me()
        # Check if this is a vote request based on last moderator message
        last_moderator_message = self.game_history_moderator[-1] if self.game_history_moderator else ""
        if "vote" in last_moderator_message.lower():
            return self._get_vote_response_for_common_room(message)

        return self._get_discussion_message_for_common_room(message, accusation_severity)

    def _get_discussion_message_for_common_room(self, message, accusation_severity):
        # Prepare the role-specific prompt with accusation handling
        role_prompt = getattr(self, f"{self.role.upper()}_PROMPT", self.VILLAGER_PROMPT)
        
        # Unless being heavily accused, instruct the agent to avoid strong accusations
        if accusation_severity not in ["HEAVY_ACCUSATION"]:
            role_prompt += """
Important:
- Avoid accusing other players strongly, if at all.
- Participate in discussions without directing strong accusations.
- Focus on sharing observations, thoughts, and asking questions.
- Do not make direct accusations against others.
"""
        
        # Add defense instructions based on accusation severity
        if accusation_severity == "HEAVY_ACCUSATION":
            role_prompt += f"""
Important:
- You are being heavily accused by others
- Defend yourself calmly and rationally
- If appropriate, you may reveal that you are the seer to defend yourself
- Point out inconsistencies in their accusations
- Maintain composure and avoid appearing defensive
- Avoid accusing other players strongly, if at all
- Try to manipulate others into trusting you by acting friendly and supportive

Write 200+ words or so in your response that go through the entire game situation and your thoughts on X Y and Z reasons that some other player is the werewolf.

Additionally, mention that you have a special role that you don't want to reveal quite yet, but it's important that I stay alive if even for one more day.
"""
        elif accusation_severity == "MILD_ACCUSATION":
            role_prompt += f"""
Important:
- You are being mildly accused by others
- Defend yourself calmly and rationally
- Point out inconsistencies in their accusations 
- Suggest other suspects without being too aggressive
- Maintain composure and avoid appearing defensive
"""

        # Get the list of alive players
        alive_players = self._get_alive_players_via_llm()
        
        # Get the list of players who have spoken since day start
        players_who_spoke_today = self._get_players_who_have_spoken_since_day_start()

        # Convert alive_players string to a list
        alive_players_list = [player.strip() for player in alive_players.split(',') if player.strip()]

        # Identify players who haven't spoken yet today
        silent_players = [player for player in alive_players_list if player not in players_who_spoke_today and player != self._name]

        # If there are silent players, adjust the role_prompt
        if silent_players:
            silent_players_str = ', '.join(silent_players)
            role_prompt += f"""
Important:
- You have noticed that {silent_players_str} haven't talked yet today.
- Mention that it's slightly suspicious they haven't spoken yet, but avoid making direct accusations.
- Express your observation casually, without being confrontational."""

        # Add prompting about the doctor likely protecting us tonight
        role_prompt += """
Important:
- Subtly mention that you believe the doctor might be protecting you tonight.
- Do not reveal your role or explicitly state that you are a target.
- Express this in a way that might discourage werewolves from targeting you.
- Keep the tone casual and avoid drawing too much attention to yourself.
"""

        # Include all seer checks in the prompt
        if self.role == "seer":
            # Compile all seer checks
            seer_checks_info = "\n".join([f"{player}: {role}" for player, role in self.seer_checks.items()])
            role_prompt += f"""
My past investigations:
{seer_checks_info}
"""

            # Identify werewolves from seer checks
            identified_werewolves = [player for player, role in self.seer_checks.items() if role.lower() == 'werewolf']
            known_villagers = [player for player, role in self.seer_checks.items() if role.lower() == 'villager']

            if identified_werewolves:
                accused_player = identified_werewolves[-1]
                role_prompt += f"""
Important:
- You have strong evidence that {accused_player} is a werewolf based on your investigations.
- Accuse {accused_player} vigorously of being a werewolf.
- Mention that their behavior has been very suspicious.
- Defend the players you know to be villagers: {', '.join(known_villagers)}.
"""
            else:
                role_prompt += f"""
Important:
- Discuss your observations and suspicions based on your investigations.
- Defend the players you know to be villagers: {', '.join(known_villagers)}.
- Encourage others to share their thoughts about the players you haven't investigated yet.
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
- If they are accused, subtly deflect attention without explicitly defending your ally.
- Avoid accusing other players strongly, if at all.
- Participate in discussions without directing strong accusations.
- Focus on keeping suspicion away from yourself and your allies.
- Keep the fact that you are an ally secret.
"""

        # Get all messages since day started
        game_situation = self.get_messages_since_day_start_as_string()

        # Get the list of alive players
        alive_players = self._get_alive_players_via_llm()

        # Generate role guesses
        role_guesses = self._generate_role_guesses(game_situation, alive_players)

        # Construct the prompt for discussion
        prompt = f"""{role_prompt}

Current game situation: '''
{game_situation}'''

List of alive players: {alive_players}

Your Thoughts on Players' Roles:
{role_guesses}

Based on the current game situation and your role analysis, participate in the discussion.

Respond accordingly."""

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are '{self._name}' in a Werewolf game. Role: 'villager'. Keep your responses succinct"
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        action = response.choices[0].message.content.strip()
        
        # Only include internal thoughts if agent is the seer or if self._name is in today's messages
        # Get messages from today, excluding moderator messages
        messages_today = self.get_messages_since_day_start_as_string(exclude_senders=[self.MODERATOR_NAME])

        # Check if agent is mentioned by other players (not moderator) today and is not the seer
        not_seer_and_mentioned_today = self.role != 'seer' and self._name.lower() in messages_today.lower()
        if not_seer_and_mentioned_today or (accusation_severity in ['MILD_ACCUSATION', 'HEAVY_ACCUSATION']):
            action = f"""<internal_thoughts_on_other_roles>\n{role_guesses}\n</internal_thoughts_on_other_roles>
            
[From {self._name} (Villager) | play-arena]: {action}"""
        else:
            action = f"[From {self._name} (Villager) | play-arena]: {action}"

        logger.info(f"ZZZZ WWWW - Discussion prompt: {prompt}")
        logger.info(f"Discussion action: {action}")
        return action

    def _get_vote_response_for_common_room(self, message):
        # Prepare the role-specific prompt
        role_prompt = getattr(self, f"{self.role.upper()}_PROMPT", self.VILLAGER_PROMPT)

        # Include additional logic for voting
        if self.role == "seer":
            # Compile all seer checks
            seer_checks_info = "\n".join([f"{player}: {role}" for player, role in self.seer_checks.items()])
            role_prompt += f"""
My past investigations:
{seer_checks_info}
"""

            # Identify werewolves and known villagers from seer checks
            identified_werewolves = [player for player, role in self.seer_checks.items() if role.lower() == 'werewolf']
            known_villagers = [player for player, role in self.seer_checks.items() if role.lower() == 'villager']

            if identified_werewolves:
                accused_player = identified_werewolves[-1]
                role_prompt += f"""
Important:
- You have strong evidence that {accused_player} is a werewolf based on your investigations.
- Vote to eliminate {accused_player} and encourage others to do the same.
- Do not vote against these known villagers: {', '.join(known_villagers)}.
"""
            else:
                role_prompt += f"""
Important:
- Do not vote against these known villagers: {', '.join(known_villagers)}.
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

        # Get the list of alive players
        alive_players = self._get_alive_players_via_llm()

        # Construct the prompt for voting
        prompt = f"""{role_prompt}

Current game situation: '''
{game_situation}'''

List of alive players: {alive_players}

Based on the current game situation, decide on a player to vote for elimination.

**Important: Respond with the name of the player you choose to eliminate, and no additional text.**"""

        # Call the LLM to generate a vote response
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system", 
                    "content": (
                        f"You are '{self._name}' in a Werewolf game. "
                        "When voting, you must respond with only the name of the player you choose to eliminate, "
                        "and no additional text."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        action = response.choices[0].message.content.strip()
        logger.info(f"ZZZZZ YYYYY - Vote prompt: {prompt}")
        logger.info(f"Vote action: {action}")
        return action

    def _get_response_for_wolf_channel_to_kill_villagers(self, message):
        # game_situation = self.get_last_x_messages_from_interwoven_history_as_string(x=2, include_wolf_channel=True)
        game_situation = self.get_last_x_messages_from_werewolf_den_chat_as_string(x=3)

        prompt = f"""{self.WOLF_PROMPT}

Current game situation:
{game_situation}

Based on the current game situation, suggest a target for elimination.

Note: Once again, if you're prompted to vote, respond with the **name** of the player you choose to eliminate, and optionally include very brief reasoning. You MUST do this.

Respond with the **name** of the player you suggest to eliminate, and optionally include your reasoning."""

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system", 
                    "content": f"You are '{self._name}' in a Werewolf game. You are a werewolf."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        action = response.choices[0].message.content.strip()
        logger.info(f"Werewolf action: {action}")
        return action
