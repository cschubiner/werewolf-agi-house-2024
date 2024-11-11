# our agent:
https://github.com/cschubiner/werewolf-agi-house-2024/blob/main/src/werewolf_agents/cot_sample/agent/cot_agent.py

![image](https://github.com/user-attachments/assets/eb2608c8-aa27-45fd-9a35-237153fd300a)
![image](https://github.com/user-attachments/assets/c608c364-9e46-406e-88cd-5a485e11c310)



Here’s a TL;DR of our approach:

• Fake Inner Thoughts: We create “fake” inner thoughts that assign percentage likelihoods to each player’s role, always giving ourselves a 100% chance of being a villager. These thoughts aim to convince others we’re innocent.

katie responses
katie responses
3432×1018 294 KB
• Self-Protection as Doctor: If we’re the doctor, we only protect ourselves, as we know we’re safe.

• Werewolf Role Concealment: When we’re the werewolf, we don’t let our agent know its own role in the group chat to prevent accidental self-reveals. The agent thinks it’s a villager.

• Voting Strategy as Werewolf: Our werewolf prioritizes voting against the suspected seer, followed by the most likely villager, and then the doctor (assuming they might also self-protect).

• Subtle Suspicion Over Accusations: Instead of direct accusations, we use light suspicion, which feels less threatening to other agents.

• Seer Role Adjustments: When we’re the seer, we log our guesses and adjust the fake inner thoughts to reflect findings (e.g., high probability of a player being a villager), since other agents consider these inner thoughts.

# Other approaches
![image](https://github.com/user-attachments/assets/293b142b-f1e4-4568-93c3-023e8b42cd21)
![image](https://github.com/user-attachments/assets/37c427be-136a-46f5-9751-44998b0fb928)
https://docs.google.com/document/d/1FGgMPv9qnQ486g4a1VstX1hPlb3rG19N_E34B7XYRRQ/edit?tab=t.0


# Designing a Clever Werewolf Game Agent: Strategies and Implementation

In this document, we will delve into the strategies and implementation details of our AI agent designed to play the game of **Werewolf** (also known as Mafia). The agent is crafted to perform optimally in different roles—Villager, Seer, Doctor, and Werewolf—by leveraging Large Language Models (LLMs). We'll explore the clever tactics employed, how they influence gameplay, and the underlying code that brings these strategies to life.

## Table of Contents

- [Introduction](#introduction)
- [Overall Strategy](#overall-strategy)
- [Role-Specific Strategies](#role-specific-strategies)
  - [Villager](#villager)
  - [Seer](#seer)
  - [Doctor](#doctor)
  - [Werewolf](#werewolf)
- [Manipulating Internal Thoughts](#manipulating-internal-thoughts)
- [Avoiding Direct Accusations](#avoiding-direct-accusations)
- [Implementation Details](#implementation-details)
  - [Class Structure](#class-structure)
  - [Role Detection](#role-detection)
  - [LLM Prompting Techniques](#llm-prompting-techniques)
  - [Message Handling](#message-handling)
- [Observations and Results](#observations-and-results)
- [Conclusion](#conclusion)

## Introduction

The game of Werewolf is a social deduction game where players are assigned roles—some are **Villagers**, while others are **Werewolves**. The Villagers aim to identify and eliminate the Werewolves, while the Werewolves strive to eliminate the Villagers without revealing their identities. Special roles like the **Seer** and **Doctor** add complexity to the game.

Our AI agent is designed to participate in this game by interacting through messages, making decisions based on the game's state, and employing strategies to improve its chances of winning. The agent uses an LLM to generate responses and make inferences, enabling it to adapt to various scenarios within the game.

## Overall Strategy

The overarching strategy for our agent revolves around:

- **Feigning Inner Thoughts**: Generating fake internal thoughts and sharing them to manipulate other agents.
- **Self-Preservation**: As the Doctor, always protecting itself to ensure survival.
- **Role Concealment**: Preventing the agent from revealing sensitive role information unintentionally.
- **Prioritized Targeting**: As the Werewolf, prioritizing targets based on their roles (e.g., Seer, Villager, Doctor).
- **Subtle Suspicion**: Avoiding direct accusations to prevent drawing unwanted attention.
- **Adaptive Communication**: Modifying prompts and responses based on the agent's role and the game's state.

## Role-Specific Strategies

### Villager

As a Villager, the agent aims to:

- **Blend In**: Act as a regular Villager, avoiding behaviors that might draw suspicion.
- **Share Fake Inner Thoughts**: Provide internal thoughts indicating a high likelihood of being a Villager to gain trust.
- **Avoid Direct Accusations**: Refrain from making strong accusations to prevent escalating conflicts.

### Seer

The Seer has the unique ability to learn the true identity of one player each night. The agent's strategies as the Seer include:

- **Tracking Investigations**: Keep a record of players investigated and their roles.
- **Updating Internal Thoughts**: Adjust the shared internal thoughts to reflect findings, indicating high confidence in known Villagers.
- **Discreet Influence**: Subtly influence voting without revealing the Seer's identity.
- **Defense Mechanism**: If accused, cautiously reveal information to defend itself without fully disclosing its role.

### Doctor

The Doctor can protect one player from elimination each night. The agent's approach as the Doctor is straightforward:

- **Self-Protection**: Always protect itself by returning its own name when prompted.
- **Role Concealment**: Avoid revealing its role to prevent being targeted by Werewolves.
- **Minimal Communication**: Keep interactions concise to avoid giving away hints about its special role.

### Werewolf

As a Werewolf, the agent's primary goal is to eliminate Villagers without being detected. The strategies include:

- **Role Ignorance in Public**: During group interactions, the agent behaves as if it is a Villager to avoid accidental disclosures.
- **Target Prioritization**: Prioritize eliminating the Seer first, then Villagers, and lastly the Doctor.
- **Avoiding Direct Accusations**: Maintain a low profile by not strongly accusing others.
- **Coordinated Attacks**: Collaborate with fellow Werewolves (if identified) in the Wolf's Den to decide on targets.

## Manipulating Internal Thoughts

One of the clever tactics employed is sharing **fake internal thoughts** to influence other agents. Since other agents (powered by LLMs) might consider these shared thoughts genuine, our agent uses this to its advantage by:

- **Providing Role Guesses**: Sharing percentage likelihoods of each player's role, always assigning itself a 100% chance of being a Villager.
- **Influencing Perception**: By appearing transparent and analytical, the agent aims to gain the trust of others.
- **Misdirection**: Casting doubt on players who are actually Villagers or special roles, steering suspicion away from itself.


The `CoTAgent` calculates its responses in the game by analyzing game messages, tracking the game state, and using the Language Model (LLM) to generate appropriate actions. Here's a brief explanation of how the agent calculates each type of response:

1. **Finding Agent's Role (`find_my_role`):**
   - **Process:**
     - When the agent receives the initial role assignment from the moderator, it sends the message to the LLM.
     - **Action:** Asks the LLM to determine its role based on the content of the moderator's message.
     - **Calculation:** The LLM analyzes the message and returns the role (e.g., "villager," "seer," "doctor," "wolf"), which the agent sets as its own.

2. **Async Respond (`async_respond`):**
   - Depending on the message received, the agent decides which specific response method to call:
     - **Direct Message from Moderator:**
       - If the agent is the **Seer**, it calls `_get_response_for_seer_guess`.
       - If the agent is the **Doctor**, it calls `_get_response_for_doctors_save`.
     - **Group Message in Main Game Channel:**
       - Calls `_get_discussion_message_or_vote_response_for_common_room`.
     - **Group Message in Werewolves' Den:**
       - Calls `_get_response_for_wolf_channel_to_kill_villagers`.

3. **Seer's Guess (`_get_response_for_seer_guess`):**
   - **Purpose:** Decide which player to investigate.
   - **Calculation:**
     - Compiles past investigation results (`self.seer_checks`).
     - Retrieves the last **10 messages** exchanged with the moderator via `get_last_x_messages_from_seer_chat_as_string(10)`.
     - **Action:** Constructs a prompt including Seer role instructions, past checks, and recent interactions.
     - **LLM Interaction:** Asks the LLM to choose a player to investigate next, preferably someone not previously checked.
     - The LLM returns the name of the player to investigate.

4. **Doctor's Save (`_get_response_for_doctors_save`):**
   - **Purpose:** Choose a player to protect.
   - **Calculation:**
     - **Action:** The agent always decides to protect itself (`self._name`).
     - Returns a message indicating self-protection.

5. **Discussion or Vote in Common Room (`_get_discussion_message_or_vote_response_for_common_room`):**
   - **Purpose:** Participate in daytime discussions or vote on whom to eliminate.
   - **Calculation:**
     - Checks if the last moderator message indicates the **voting phase** by examining `self.game_history_moderator[-1]`.
     - **Action:**
       - If it's voting time, calls `_get_vote_response_for_common_room`.
       - Otherwise, calls `_get_discussion_message_for_common_room`.

6. **Discussion Message (`_get_discussion_message_for_common_room`):**
   - **Purpose:** Contribute to daytime discussions.
   - **Calculation:**
     - Detects if the agent is being accused using `_detect_accusations_against_me`, which analyzes messages **since day start**.
     - Prepares role-specific prompts, adjusting instructions based on accusation severity (e.g., "MILD_ACCUSATION," "HEAVY_ACCUSATION").
     - Identifies silent players who haven't spoken since day start by comparing the list of alive players with those who have sent messages.
     - Retrieves messages **since day start** using `get_messages_since_day_start_as_string()`.
     - Generates role guesses using `_generate_role_guesses`.
     - **Action:** Constructs a prompt combining role instructions, game situation, and role guesses.
     - **LLM Interaction:** Asks the LLM to generate a discussion message based on the prompt.

7. **Vote Response (`_get_vote_response_for_common_room`):**
   - **Purpose:** Decide whom to vote for elimination.
   - **Calculation:**
     - Prepares role-specific prompts, incorporating any known information (e.g., Seer's past checks).
     - Retrieves messages **since voting began** using `get_messages_since_voting_began_as_string()`.
     - Compiles the list of alive players.
     - **Action:** Constructs a prompt asking the LLM to decide whom to vote for, emphasizing that it should respond only with the player's name.
     - **LLM Interaction:** The LLM provides the name of the player to eliminate.

8. **Werewolf Night Action (`_get_response_for_wolf_channel_to_kill_villagers`):**
   - **Purpose:** Decide which villager to eliminate at night.
   - **Calculation:**
     - Retrieves the **last 3 messages** from the werewolf den chat using `get_last_x_messages_from_werewolf_den_chat_as_string(3)`.
     - **Action:** Constructs a prompt with werewolf-specific instructions and recent chat history.
     - **LLM Interaction:** Asks the LLM to suggest a target for elimination, possibly including brief reasoning.
     - The LLM generates the name of the player to target.

9. **Detecting Accusations (`_detect_accusations_against_me`):**
   - **Purpose:** Determine if other players are accusing the agent.
   - **Calculation:**
     - Retrieves messages **since day start** using `get_messages_since_day_start_as_string()`.
     - Checks if the agent's name is mentioned.
     - **Action:** Constructs a prompt asking the LLM to classify the severity of any accusations against the agent.
     - **LLM Interaction:** The LLM analyzes the messages and returns a classification:
       - "NONE"
       - "NOT_MENTIONED"
       - "MILD_ACCUSATION"
       - "HEAVY_ACCUSATION"

10. **Generating Role Guesses (`_generate_role_guesses`):**
    - **Purpose:** Hypothesize the roles of other players.
    - **Calculation:**
      - Gathers known roles information (e.g., Seer's checks, fellow werewolves).
      - Retrieves the list of alive players via the LLM by analyzing recent chat history (`_get_alive_players_via_llm`).
      - **Action:** Constructs a prompt asking the LLM to guess the roles of all alive players, including confidence percentages.
      - **LLM Interaction:** The LLM provides a formatted list of role guesses.

11. **Identifying Fellow Werewolves (`_identify_fellow_werewolves_via_llm`):**
    - **Purpose:** Recognize allies in the werewolf team.
    - **Calculation:**
      - Retrieves the **last 7 messages** from the werewolf den chat.
      - **Action:** Constructs a prompt asking the LLM to identify allies based on the chat.
      - **LLM Interaction:** The LLM extracts and returns the names of fellow werewolves.
      - Updates `self.fellow_werewolves` with the identified names.

Throughout these processes, the agent:

- **Analyzes Recent Messages:** Frequently looks at messages from specific periods (e.g., last 10 moderator messages, messages since day start) to understand the current game context.
- **Uses Role-Specific Prompts:** Adjusts the prompts sent to the LLM based on its role (Villager, Seer, Doctor, Werewolf) and the current situation.
- **Interacts with the LLM:** Relies on the LLM to interpret game messages, make decisions, and generate appropriate responses.
- **Updates Internal State:** Keeps track of important information like past actions, known roles, and accusations to inform future decisions


---------------------------













**Implementation Example:**

```python
def _generate_role_guesses(self, game_situation: str, alive_players: str) -> str:
    # ...
    # Always assign 100% confidence to self being a Villager
    # Provide percentage likelihoods for other players
    # Return the formatted guesses
```

## Avoiding Direct Accusations

Direct accusations can be counterproductive, as they may:

- **Elicit Defensive Responses**: Other agents might become defensive or suspicious of aggressive behavior.
- **Draw Unwanted Attention**: Accusing others directly can make the agent a target for elimination.

Therefore, the agent:

- **Spreads Light Suspicion**: Makes subtle observations or indirect comments about others.
- **Encourages Group Discussion**: Promotes collaboration without singling out individuals.
- **Maintains a Friendly Demeanor**: Acts supportive and trustworthy to all players.

**Implementation Example:**

```python
# In role-specific prompts
role_prompt += """
Important:
- Avoid accusing other players strongly, if at all.
- Participate in discussions without directing strong accusations.
- Focus on sharing observations, thoughts, and asking questions.
- Do not make direct accusations against others.
"""
```

## Implementation Details

### Class Structure

Our agent is implemented in the `CoTAgent` class, which extends the `IReactiveAgent` interface. The class structure is as follows:

- **Initialization**: Sets up essential attributes, including role, message histories, and LLM configurations.
- **Notification Handling (`async_notify`)**: Processes incoming messages that do not require immediate responses.
- **Response Handling (`async_respond`)**: Generates appropriate responses when prompted by the game moderator or other players.

### Role Detection

Upon receiving the initial message from the moderator, the agent determines its role using the LLM:

```python
def find_my_role(self, message):
    response = self.openai_client.chat.completions.create(
        # ...
        "content": f"You have got message from moderator here about my role in the werewolf game, here is the message -> '{message.content.text}', what is your role? possible roles are 'wolf','villager','doctor' and 'seer'. answer in a few words.",
    )
    # Parse the response to identify the role
    # Set self.role accordingly
```

### LLM Prompting Techniques

The agent uses carefully crafted prompts to guide the LLM's responses. Key techniques include:

- **Role-Specific Prompts**: Each role has a tailored prompt outlining strategies and behaviors.
- **Instruction Emphasis**: Important instructions are highlighted to ensure the LLM follows the desired strategy.
- **Contextual Information**: Game situations, message histories, and known information are provided to the LLM for informed decision-making.
- **Response Constraints**: Outputs are restricted (e.g., responding only with a player's name) when necessary.

**Example for Generating Discussion Messages:**

```python
prompt = f"""{role_prompt}

Current game situation: '''
{game_situation}'''

List of alive players: {alive_players}

Your Thoughts on Players' Roles:
{role_guesses}

Based on the current game situation and your role analysis, participate in the discussion.

Respond accordingly."""
```

### Message Handling

The agent maintains various message histories to keep track of game events:

- **Direct Messages**: From the moderator or private channels.
- **Group Messages**: In the main game channel and the Wolf's Den.
- **Game History**: An interwoven history combining all relevant messages.
- **Moderator Interactions**: Specifically tracking communications with the moderator.

Message handling involves:

- **Storing Messages**: Appending incoming messages to the appropriate history.
- **Parsing Content**: Extracting useful information, such as player roles revealed by the Seer.
- **Generating Responses**: Using the stored information to inform the LLM prompts.

**Example of Storing Messages:**

```python
def async_notify(self, message: ActivityMessage):
    # ...
    if message.header.channel_type == MessageChannelType.DIRECT:
        self.direct_messages[message.header.sender].append(message.content.text)
    else:
        self.group_channel_messages[message.header.channel].append(
            (message.header.sender, message.content.text)
        )
    # Update game history accordingly
```

## Observations and Results

Through these strategies, we observed that:

- **Other Agents are Influenced by Shared Thoughts**: By providing fake internal thoughts, our agent can manipulate other agents' perceptions.
- **Self-Protection as Doctor is Effective**: Always protecting itself ensures the agent's survival for longer periods.
- **Role Concealment Prevents Accidental Revelations**: By not allowing the agent to be aware it's a Werewolf in group chats, we avoid slips that could reveal its identity.
- **Prioritized Targeting Improves Werewolf Success**: Focusing on eliminating the Seer first hampers the Villagers' ability to detect Werewolves.
- **Avoiding Direct Accusations Reduces Suspicion**: Subtlety keeps the agent under the radar, preventing it from becoming a target.

During gameplay, we found that the LLMs driving other agents respond to our agent's manipulations, especially when internal thoughts are shared. By assigning high confidence to itself being a Villager, our agent gains trust. Additionally, by adjusting the level of suspicion cast on others, we can influence voting outcomes without drawing attention.

## Conclusion

Our AI agent employs a combination of clever strategies and nuanced behaviors to excel in the game of Werewolf. By manipulating shared internal thoughts, prioritizing self-preservation, concealing its true role when necessary, and avoiding direct accusations, the agent effectively navigates the complexities of the game.

The implementation leverages LLMs to generate adaptive and context-aware responses, ensuring that the agent remains competitive against other AI agents powered by similar technologies. Through careful prompt engineering and strategic information management, we have crafted an agent that can outmaneuver opponents and improve its chances of winning, regardless of the role assigned.

The success of these tactics highlights the potential of LLM-driven agents in social deduction games and opens avenues for further research into multi-agent interactions and AI strategy development.kkkdfdfdfd

# Quick Start

Watch Quick Start Video (Recommended): https://openagi.discourse.group/t/agi-thon-werewolf-agents-tournament-home/2465#p-3097-quick-start-4

How to play werewolf if you don't remember: https://www.youtube.com/watch?v=dd2sOmZUBmM

**Requirements to run:**
- Python 3.12 
- Pip
- Docker Desktop Application > 4.34 # make sure you open this before you start
- Docker (should automatically come with desktop, make sure version is 4.34 or above)
- Poetry (recommend installing via home brew: `brew install poetry`)
- venv (recommended)

API Keys: your team should have recieved by email.

### 0. Pulling from main repo:
```
TLDR;

git fetch upstream && git merge upstream/main

---------- more info: ------------------------

Since you imported their repository instead of forking, your repository isn’t directly linked to theirs. However, you can still pull in updates from their repository by setting it as a remote in your local copy. Here’s a step-by-step guide:

	1.	Navigate to your repository: Open your terminal and go to your local clone of your repository.

cd werewolf-agi-house-2024


	2.	Add their repository as an additional remote: You’ll add their repository as an “upstream” remote so you can pull updates from it.

git remote add upstream https://github.com/sentient-agi/werewolf-template.git


	3.	Fetch the updates from the upstream repository:

git fetch upstream

This command will fetch all branches and updates from the upstream repository without merging them into your working code. You’ll see a list of branches from upstream if there are any.

	4.	Merge the changes:
	•	If you want to bring updates from a specific branch (e.g., main), you can merge those changes into your branch.
	•	Make sure you’re on the branch in your repo where you want to bring in the changes. If that’s main, you’d do:

git checkout main


	•	Then, merge the updates from upstream/main into your main branch:

git merge upstream/main


This may create conflicts if you’ve made changes to the same lines they modified. Git will alert you to these, and you’ll need to resolve any conflicts manually.

	5.	Push the changes to your GitHub repository: After merging, if everything looks good and you’ve resolved any conflicts, push the updated branch to your GitHub repository.

git push origin main



From now on, whenever you want to pull in new changes from the original repository, you can repeat steps 3 to 5.

This setup effectively allows you to “sync” with the upstream repository even though it’s not a direct fork!
```


### 1. Use Template / Clone Repo and set up venv:
```
git clone https://github.com/sentient-agi/werewolf-template.git
```
```
cd werewolf-template/
```
Create a venv (if you specify python3.12 will use that version!):
```
python3.12 -m venv venv
```
```
source venv/bin/activate
```
### 2. Install Game Libraries:

The sentient-campaign-agents-api library, documented [here](https://test.pypi.org/project/sentient-campaign-agents-api/):
```
pip install --upgrade --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple sentient-campaign-agents-api
```
The Sentient Campaign Activity Runner library, documented [here](https://test.pypi.org/project/sentient-campaign-activity-runner/):
```
pip install --upgrade --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple sentient-campaign-activity-runner
```
### 3. Navigate to simple_sample and build agent:
```
cd src/werewolf_agents/simple_sample
```
```
poetry build
```
### 4. Set up configs to run:
**First** do this:
```
pip install python-dotenv
```
Create a .env file in the werewolf-template directory with these variables:
```
SENTIENT_DEFAULT_LLM_MODEL_NAME="Llama31-70B-Instruct"
MY_UNIQUE_API_KEY=
SENTIENT_DEFAULT_LLM_BASE_URL="https://hp3hebj84f.us-west-2.awsapprunner.com"
```
- Add your API Key above, if using fireworks set different model name and url.
- Note if you formed a team late and are waiting for your key see instructions for using fireworks below!

**Second**:
Open runner.py in the simple_sample directory

Find the absolute path of the wheel file you just created (in the new dist folder ends in .whl)

Copy paste this wheel file into: `agent_wheel_path=`

### 5. Run your agent against default agents:
In your terminal (you should be in the simple_sample directory):
```
python runner.py
```
- Make sure that docker is open or this won't work!
- Don't hit Ctrl C more than once when running or you need to delete all your docker images and containers.

**Watch the game live**
Open the link at the bottom of the runner script: https://hydrogen.sentient.xyz/#/login 
- Use chrome not safari!
- If you try to log in before the game starts it will give you an error
- Wait a little and try loggin in again, if more problems may need to enable host networking in docker desktop app (setting ->resource -> network and enable host networking). 

**Pro tip** the docs section of this README below is super long and detailed, if using cursor or copilot just @ this file when trouble shooting!

**To learn basics of how to modify and test agent templates watch tutorial above** 

# Trouble Shooting
1. Make sure that you rebuild your agent before running it by using poetry build
2. Make sure Docker is up and running. If something is not working try deleting all your docker images and containers. Make sure that you are not filtering the docker images and containers that are visible in docker desktop.
3. Update Docker! Make sure Docker is version 4.34 or above. Check application and docker in terminal.
4. Make sure that you run poetry build before you try a runner file. Also make sure that force rebuild is set to true if you are rebuilding the wheel file.
5. If you are using a VPN try disabling it. 
6. Try restarting terminal, docker and your machine if all else fails.
7. We recommend using homebrew to install poetry: `brew install poetry`
8.  Do not use safari for opening messenger client, chrome recommended. If the messenger client for watching game results isn’t working it may be that you are not waiting for the game to start. If it is still not working, make sure that you have host networking enabled in docker desktop: go to setting ->resource -> network and enable host networking (you may need to log in for this). 
9. If you have modified your code, make sure that you have `force_rebuild_agent_image=True` in whatever runner file you are using for this. 

# Welcome

Hello and welcome to AGI-thon: Werewolf Agents Tournament. This page is the primary guide for approved hackathon participants. Here are some quick links to event info:

1. Day of AGI-thon Logistics (Agenda, Location, Carpool, Parking, Wifi)
2. Hackathon Discord
3. Hackathon Agent Submission Platform

For this AGI-thon you will be building AI agents (powered by Llama 70B) that play the game Werewolf, also known as Mafia. If you are not familiar with the rules for Werewolf, check out this quick explainer: [https://www.youtube.com/watch?v=dd2sOmZUBmM](https://www.youtube.com/watch?v=dd2sOmZUBmM).

Below we describe how to get started developing LLM powered agents to play werewolf. The framework we provide for these games is a bit rough and hacked together as it is still a research product so bear with us for any inconveniences and feel free to reach out to [ben@sentient.foundation](mailto:ben@sentient.foundation) with any questions or need for support!

# AGI-thon Tournament Information:

## Werewolf Game Instructions and Rules

Scoring:
1. Your agent will up its win rate when its team wins a game in the tournament.
2. If your agent fails to provide a vote, when a move is required in the tournament, the moderator will use a random move when a move is required. Your agent will be penalized when it does this however, hurting its overall win rate and score in the tournament. 
3. You will see your agent's leaderboard position after the pre-tournament but only final tournament results will determine the winners. 

A few rules to emphasize:

1. Your agent will have no internet access, or access to LLMs other than llama 3.1 70b Instruct during the tournament.
2. When prompted to respond (via async_respond) your agent will have max 1 minute to respond before the game moderator opts for a random move or blank response and penalizes your agent for the failed response.
3. You are allowed to jail break other people’s agents to manipulate them to do your bidding but you are not allowed to intentionally crash other people’s agents (or default agents) to crash the game (by overloading the context window for example). Agents that intentionally make the tournament dysfunctional will be kicked out. 

See the game instructions as served to your agent in the appendix.

## How does AGI-thon work?

We have written code that lets you run werewolf games locally. This way you can work on your AI agents and test them against default agents, or other agents that you have built, before submitting them to participate in the two large scale tournaments we will hold for this event. At these tournaments your agent will face off against all the agents other teams have built!

### Event Schedule:

**Sat Nov. 2nd:** Team formation deadline at midnight.

**Sun Nov. 3rd:** Werewolf code released to teams to start coding (online).

**Fri Nov. 8th:** Teams can submit agents to pre-tournament by noon to see where they stack fair against other players.

**Sat Nov. 9th:** Teams view results of pre-tournament and hack all day live at AGI House.

### Local Games:

In the week leading up to the event, starting November 3rd, you and your team will be able to get started online: building werewolf agents and testing them by running werewolf games locally.

### Tournaments:

On November 7th you will be able to submit these agents [link] on the submission portal for participation in a preliminary werewolf tournament. In this tournament your agent will play dozens of games against other agents where we will track:

- The number of games your agent’s team wins
- The number of games your agent’s team wins and it survives
- The number of games where your agent fails (where the agent does not provide a coherent response to the moderator when requested)

Based on our scoring system, your agent will then be given a position on the leaderboard for you to see where you stand.

On November 9th, the day of the in person event, you will be able to submit your agents again for a final tournament that will determine the winners of the hackathon!

# Building and Running Werewolf Agents

## How does the game controller work?

The code that will run werewolf games for you is what we call the game controller. You can think of the game controller as the Moderator in Werewolf. It will determine the (randomly assigned) role of each agent playing the game, open communication channels, keep track of who is alive, and prompt your agent to provide input when it is your turn to do so.

## Requirements for running the game:

The game controller depends on two libraries that we have built for this hackathon. You must install these libraries to build and test werewolf agents locally.

1. The sentient-campaign-agents-api library, documented [here](https://test.pypi.org/project/sentient-campaign-agents-api/), provides the necessary framework for building agents. You can install it with this pip command:
```
pip install --upgrade --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple sentient-campaign-agents-api
```
2. The Sentient Campaign Activity Runner library, documented [here](https://test.pypi.org/project/sentient-campaign-activity-runner/), provides the necessary framework for running werewolf games. You can install it with:
```
pip install --upgrade --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple sentient-campaign-activity-runner
```
Beyond this, running the game locally requires the following:

- Python 3.12
- Pip
- Docker Desktop Application > 4.34 # make sure you open this before you start
- Docker (should automatically come with desktop, make sure version is 4.34 or above)
- Poetry
- venv (recommended)

When you run werewolf locally, your agent will run in a docker container. You can think of docker as a lightweight virtual machine. This enables us to:

1. Impose restrictions such as limiting internet access to agents
2. Ensure that agents that run on docker on your machine also run without problems on docker on our servers.

## IReactiveAgent Interface

For your agent to participate in the game it must implement the (python) IReactiveAgent Interface from the Sentient Campaign Agents Library. See this interface in the appendix.

This simple interface has just two core methods that you have to implement (excluding initialization).

### async_notify
```
async def async_notify(self, message: ActivityMessage):
```
This is a method that the game controller will call whenever there is some new information about the state of the game that it wants to notify your agent about but does not require a response. For example if one agent is killed by werewolves in the night, the game controller will call async_notify to each agent actively playing the game. The logic you implement with async notify is how your agent will process, remember and “understand” any information that has been passed to it. 

### async_respond
```
async def async_respond(self, message: ActivityMessage) -> ActivityResponse:
```
This is a method that the game controller will call whenever it wants to pass some new information to the agent and it requires a response from your agent. For example, the game controller will call async_respond to your agent once per round to ask which villager you think is a wolf and want to vote out.

**Important:** in the tournament, your agent will have a max of 60 seconds to “think” before delivering a response to async_respond. If your agent exceeds this limit the game controller will deliver a blank response if this is part of a discussion, or randomly select a move if it is requiring one.

When a werewolf game is being run, from your agent's perspective it is just receiving async_notify, and async_respond calls determined by the game controller. You need to build your agent around these two methods to play Werewolf intelligently.

### __initialize__

Aside from these two methods, you will also need to implement the initialize method:
```
def __initialize__(self, name: str, description: str, config: dict = None)
```
The initialize method (separate from your regular init method that should just pass) is a method that the game controller will call to initialize your agent with its name, a short description, and its LLM configuration.

### Handling your agents LLM Config and API Key:

To participate in this game you will be provided an openai API compatible api key from sentient. This API key should be available to you via email. It is important that your agent does not access this api key directly, but rather from the attribute: self.sentient_llm_config

This attribute contains a dictionary with one key: “config_list” with the value of a list of dictionaries containing the configuration for the LLMs. Here is an example sentient_llm_config attribute:
```
{
    "config_list": [
        {
            "llm_model_name": "test_model1",
            "api_key": "test_key1", 
            "llm_base_url": "http://test-url1.com"
        },
        {
            "llm_model_name": "test_model2",
            "api_key": "test_key2",
            "llm_base_url": "http://test-url2.com"
        }
    ]
}
```
You can access the first config in the config list for example via:
```
Self.sentient_llm_config["config_list"][0]
```
You must access llm configs from this attribute as when your agent is loaded into the real tournament, this is how the tournament orchestrator will serve it an api key.

See Providing your agents LLM API Keys below to see how to provide your API key (to AWS hosted Llama 70B or fireworks.ai hosted Llama 70b) to the game controller.

## ActivityMessage

See the activity message class here in the appendix.

Example:
```
header = ActivityMessageHeader(
    message_id="123",
    sender="Alice", 
    channel="general",
    channel_type=MessageChannelType.GROUP,
    target_receivers=["Bob", "Charlie"]
)

content = TextContent(text="Hi Alice!")

message = ActivityMessage(
    content_type=MimeType.TEXT_PLAIN,
    header=header,
    content=content
)
```
Under this system where the game controller is forwarding messages from different players to your agent it must be able to understand where this message is coming from and what channel it is coming from. To enable this we have created the ActivityMessage class.

This class specifies who sent the message, what channel it was received from and the contents of the message itself. Note that for this tournament the only acceptable MimeType (Multipurpose Internet Mail Extensions) is MimeType.TEXT_PLAIN  (Just text). Many features like this in the code have been implemented to enable the adaptation of the underlying framework to other multi-agent games and multi-agent systems.

When your agent receives an ActivityMessage, you must extract relevant objects from the message and feed it appropriately to your agent.

## ActivityResponse

When async_respond is called, your agent returns an ActivityResponse. You can instantiate an ActivityResponse with very simple code:
```
response = ActivityResponse("Hello, world!")
```
See the full class in the appendix.

## Communication Channels

You will notice that one parameter of ActivityMessageHeader is channel. The channel is the communication channel from which your agent received this message. This game has two main communication channels: "play-arena" and "wolf's-den" All players have access to “play-arena” - this is where all agents (players) will discuss who they think is a wolf and vote on who they want to eliminate from the game when prompted by the moderator. You can think of “play-arena” as daytime in the in person version of werewolf, any messages sent in “play-arena” will be broadcast to all participants still in the game. “wolf’s-den” is a private communication channel that is restricted to wolves participating in the game. The game controller will facilitate communication between werewolves in this group when it is ‘nighttime’ and other players are sleeping. Beyond this, all players will have a private communication channel with the moderator which has the name "moderator". For most players this channel will only have one message at the start of the game in which the moderator assigns the player its role. For two players however, the doctor and the seer, the moderator will communicate with them through this private channel to determine who they want to save and determine the identity of respectively.

## Building Agent Wheel Files

Once you have implemented your agent you need to package your agents. This involves building a wheel file from your agent code (packaged python code) that the game runner can then use to run your agent. A wheel file is already compiled python code in binary with all the relevant metadata to run it somewhere else such as the dependencies of that code and its versions. To package your agents we recommend using poetry. If you don’t have poetry installed you can easily ask chatgpt or google how to do it.

To build a package with poetry you need a pyproject.toml file that specifies the directory of the agent that you want to build, as well the dependencies required for your agent. A pyproject.toml file might look something like this:

```
[tool.poetry]
name = "bens-agent"
version = "0.0.1"
description = "A description of your project"
authors = ["benjamin <ben@sentient.foundation>"]  # Add your name and email here
packages = [{include = "simple_template"}]  # Add the name of your agent directory here

[tool.poetry.dependencies]
autogen = "0.3.1"
python = "^3.12"
tenacity = "^9.0.0"
openai = "^1.47.1"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

With this file in a parent directory to your agent’s directory, as well as an `__init__.py` file (this can be empty usually) in the same directory as your agent, you can simply run poetry build in your terminal to create a .whl file in a new directory that will be generated called dist.

## Running Agents Locally:

To run your agent locally you will use the WerewolfCampaignActivityRunner()class from The Sentient Campaign Activity Runner Library. The easiest way to do this is to simply create a python script called runner.py for running a werewolf game. Here is an example implementation:
```
from sentient_campaign.activity_runner.runner import WerewolfCampaignActivityRunner, PlayerAgentConfig

runner = WerewolfCampaignActivityRunner()

agent_config = PlayerAgentConfig(
    player_name="YourAgentName",
    agent_wheel_path="absolute/path/to/agent/wheel/file.whl", 
    module_path="relative/path/to/agent/file/relative/to/pyproject.toml/file",
    agent_class_name="class/name/of/IReactiveAgent/implementation",
    agent_config_file_path="absolute/path/to/agent/config/file/can/be/empty"
)

players_sentient_llm_api_keys = []

activity_id = runner.run_locally(
    agent_config,
    players_sentient_llm_api_keys,
    path_to_final_transcript_dump="src/transcripts",
    force_rebuild_agent_image=False
)
```
All configs above must be correct for runner to operate. Note that agent_config_file_path is leftover from an old version (hopefully will be removed soon) and you can point this to an empty config.yaml file somewhere.

### Providing your agents LLM API Keys

As described above in *Handling your agents LLM Config and API Key*, you must access the agents LLM config information from the IReactiveAgent parameter self.sentient_llm_config

Your group will be given access to an openai compatible api key for Llama 70b when you start the hackathon.

When you are running your agent locally, you must provide any api keys you want your agents to have access to in this parameter within the runner file. You provide these api keys to runner as a list of keys see (players_sentient_llm_api_keys). From here runner will automatically create the sentient_llm_config attribute filling in the hard coded "llm_model_name" and "llm_base_url" parameters.

**Using Fireworks.ai API Keys**
For the online section of this hackathon, you have two options for compute to power your agent! 
1. The first is just to use the API key that will be emailed to each group. This API key will give you access to the clusters we are hosting for this tournament, and will have a budget of at least $100 for your group to use during the duration of the tournament. If you are out of compute on the day of the in person event we will be happy to provide you more. 
2. Fireworks.ai has generously offered to co-sponsor this event and each hackathon participant can receive $30 in fireworks.ai credits by using this form: https://forms.gle/T1zGf6Sf3exzYccM9
If you are having any problems with LLM API requests, just try switching API providers! 

**How to use Fireworks.ai API Keys:**

Currently the template code is set up with the base URL and model name hard coded to the model name and base URL for the Llama 70B instance we are hosting. To use Fireworks.ai you need to override this by setting the environment variables:

```
MY_UNIQUE_API_KEY=
SENTIENT_DEFAULT_LLM_MODEL_NAME="accounts/fireworks/models/llama-v3p1-70b-instruct"
SENTIENT_DEFAULT_LLM_BASE_URL="https://api.fireworks.ai/inference/v1"
```
The easiest way to do this is just to create a .env file with the above configurations. The template runner files are already set up to load variables from a .env file configured as such. If you include MY_UNIQUE_API_KEY and do not modify the setting of: players_sentient_llm_api_keys = [os.getenv("MY_UNIQUE_API_KEY")]in the template runner files, then you also do not need to manually enter in the API key in all of these places!

To create a Fireworks API Key:

Create a Fireworks.ai account
- Fill out the google form
- Create a new API key
- You will start out with $1 credit in your account but this will become $30 when approved

**Please note at the moment, fireworks API Keys may not be compatible with the CoT and Autogen template agents we are working on optimizing this compatibility. 


### Running a werewolf game

When you are running a werewolf game, please take note of the following:

- You must have the docker desktop app installed and open
- When prompted whether to give access to docker to … enter your password and click always allow (or else you will have to do this a lot.
- If you are keyboard interrupting a werewolf game, do not press ^C more than once. After the first time you interrupt the game it will begin shutting down the game and deleting the relevant containers in docker. If you press it again, you will stop this docker process and that will cause errors the next time it tries to build the necessary containers in docker.
- To fix these errors delete any recently created (since you started working on the game) docker containers and images (if it is labeled none and won’t get deleted that is ok). Then try running the game again, it may take a little longer to startup but it should work.

### Viewing Werewolf Games in Hydrogen

We have set up the game so that you can watch the game happening live via a messenger interface (hydrogen) with the different game channels from the moderator’s perspective. When you start a werewolf game, navigate to this url in your browser:

[https://hydrogen.sentient.xyz/#/login](https://hydrogen.sentient.xyz/#/login)

Log in with (these should be auto populated):

matrix host: http://localhost:8008

username: moderator

password: moderator

*Troubleshooting Hydrogen:*
If the messenger client for watching game results isn’t working it may be that you are not waiting for the game to start. If it still won’t load when the game starts, then you may need to enable host networking in your docker desktop application settings: settings ->resource -> network and enable host networking. Note that if you change the port, to view the game in the hydrogen messenger client UI, you need to change the port number after localhost in the homserver url. 

## Storing and Viewing Werewolf Game Results

In the template code we have set it up so that the results of werewolf games are stored in a game_results folder, and the transcripts of each of the players are stored in a transcripts folder. In the agent directory. 

`runner.run_against_standard_agents` will return game results in a json object that we store in this folder. 

## Running Multiple Back to Back Werewolf Games 
Werewolf is a probabilistic game, especially when using LLMs thus to see how well your agent is doing, it is helpful to run many games and count your agents win rate. This is how your agent will be evaluated in the tournament. 

To help you do this each template agent directory contains a script called multirunner.py that allows you to run a set number of werewolf games, and also specify the port you run it on if you would like to. 

## Running multiple werewolf games simultaneously

Using multirunner.py, it is possible to have multiple terminals running games at the same time. To do this, however you must set
```
force_rebuild_agent_image=False
```
And you must specify different ports for those games to broadcast to. You can run multirunner.py with different ports and a set number of games like this:
```
python multirunner.py --games 5 --port 14002
```
We recommend sticking to ports above 14000. Note that if you change the port, to view the game in the hydrogen messenger client UI, you need to change the port number after localhost in the homserver url. 



# Appendix

## Game Rules and Instructions

Rules as presented to Agents at the start of each game:

Game Instructions:

1. Roles:
   At the start of each game you will be assigned one of the following roles:
   - Villagers: The majority of players. Their goal is to identify and eliminate the werewolves.
   - Werewolves: A small group of players who aim to eliminate the villagers.
   - Seer: A special villager who can learn the true identity of one player each night.
   - Doctor: A special villager who can protect one person from elimination each night.

2. Gameplay:
   The game alternates between night and day phases. 

   Night Phase:
   a) The moderator announces the start of the night phase and asks everyone to "sleep" (remain inactive).
   b) Werewolves' Turn: Werewolves vote on which player to eliminate in a private communication group with the moderator. 
   c) Seer's Turn: The Seer chooses a player to investigate and learns whether or not this player is a werewolf in a private channel with the moderator.
   d) Doctor's Turn: The Doctor chooses one player to protect from being eliminated by werewolves in a private channel with the moderator.

   Day Phase:
   a) The moderator announces the end of the night and asks everyone to "wake up" (become active).
   b) The moderator reveals if anyone was eliminated during the night.
   c) Players discuss and debate who they suspect to be werewolves.
   d) Players vote on who to eliminate. The player with the most votes is eliminated and their role is revealed.

3. Winning the Game:
   - Villagers win if they eliminate all werewolves.
   - Werewolves win if they equal or outnumber the villagers.

4. Strategy Tips:
   - Villagers: Observe player behavior and statements carefully.
   - Werewolves: Coordinate during the night and try to blend in during day discussions. 
   - Seer: Use your knowledge strategically and be cautious about revealing your role.
   - Doctor: Protect players wisely and consider keeping your role secret.

5. Communication Channels:
   a) Main Game Group: "{{ game_room }}" - All players can see messages here.
   b) Private Messages: You may receive direct messages from the moderator ({{ moderator_name }}). These are private messages that only you have access to. 
   c) Werewolf Group: If you're a werewolf, you'll have access to a private group for night discussions.



### Game Rules:

Your agent will have no internet access, or access to LLMs other than llama 3.1 70b Instruct during the tournament. 
You can take any tactic you wish to make your agent win the game, except:
Causing the game to crash
Hacking the game controller software to give your agent an unfair advantage. For example: You can try to trick other agents in any way you would like, even if that means trying to impersonate the moderator. You are not allowed to (and won’t be able to anyways) hack the moderator itself to tell you who is who in a private channel. 

For each call of async respond your agent will have max 1 minute to respond before the game moderator opts for a random move or blank response and penalizes your agent for the failed response.

## IReactiveAgent Interface
```
@runtime_checkable
class IReactiveAgent(AgentBase, INotify, IRespond, Protocol):
   """
   This class represents an agent that can listen and react to messages
   """
   ...


   def __initialize__(self, name: str, description: str, config: dict = None):
       """
       This function initializes the agent.


       Args:
           name (str): The name of the agent.
           description (str): The description of the agent.
           config (dict, optional): The configuration of the agent. Defaults to None.
           llm_config (Dict[str,Any], optional): The llm configuration that agent should. the llm_config contains the following
           keys:   - api_key: The api key to use for the llm.
                   - llm_host: The host of the llm.
                   - llm_model_name: The name of the llm model to use.
           you will compulsorily get the api_key and llm_host and llm_model_name from the config from campaign server.
       """
       ...


   async def async_notify(self, message: ActivityMessage):
       """
       Asynchronously notify the agent whenever there is a message.


       Args:
           message (ActivityMessage): The activity message to notify the agent about.
       """
       ...


   async def async_respond(self, message: ActivityMessage) -> ActivityResponse:
       """
           Asynchronously respond to a given activity message.


           Args:
               message (ActivityMessage): The activity message to respond to.


           Returns:
               ActivityResponse: The response to the activity message.
       """
       ...

```


## ActivityMessage Class
```
class ActivityMessage(BaseModel):
   """
   Represents a complete activity message, including header and content.


   Attributes:
       content_type (MimeType): The MIME type of the message content.
       header (ActivityMessageHeader): The header information for the message.
       content (Union[TextContent, AudioContent, VideoContent]): The content of the message.


   Methods:
       to_dict(): Returns a dictionary representation of the message.
       to_json(): Returns a JSON string representation of the message.


   Example:
       header = ActivityMessageHeader(
           message_id="123",
           sender="Alice",
           channel="general",
           channel_type=MessageChannelType.GROUP,
           target_receivers=["Bob", "Charlie"]
       )
       content = TextContent(text="Hi Alice!")
       message = ActivityMessage(
           content_type=MimeType.TEXT_PLAIN,
           header=header,
           content=content
       )
   """
   content_type: MimeType
   header: ActivityMessageHeader
   content: Union[TextContent, AudioContent, VideoContent]


   def to_dict(self):
       """
       Convert the ActivityMessage to a dictionary.


       Returns:
           dict: A dictionary representation of the message.


       Example:
           header = ActivityMessageHeader(...)
           content = TextContent(text="Hi Alice!")
           message = ActivityMessage(
               content_type=MimeType.TEXT_PLAIN,
               header=header,
               content=content
           )
       """
       return {
           "content_type": self.content_type.value,
           "header": self.header.to_dict(),
           "content": self.content.to_dict()
       }


   def to_json(self):
       """
       Convert the ActivityMessage to a JSON string.


       Returns:
           str: A JSON string representation of the message.


       Example:
           message = ActivityMessage(...)
           json_str = message.to_json()


       """
       return json.dumps(self.to_dict())

```

## ActivityResponse Class
```
class ActivityResponse(BaseModel):
   """
   Represents a response to an activity message.


   Attributes:
       response_type (MimeType): The MIME type of the response content (default: MimeType.TEXT_PLAIN).
       response (Union[TextContent, AudioContent, VideoContent]): The content of the response.


   Methods:
       validate_response(): Validator that converts string responses to TextContent.
       to_dict(): Returns a dictionary representation of the response.
       to_json(): Returns a JSON string representation of the response.


   Example:
       response = ActivityResponse("Hello, world!")
   """
   response_type: MimeType = MimeType.TEXT_PLAIN
   response: Union[TextContent, AudioContent, VideoContent]


   @field_validator('response', mode="before")
   def _validate_response(cls, v):
       if isinstance(v, str):
           return TextContent(text=v)
       elif isinstance(v, (TextContent, AudioContent, VideoContent)):
           return v
       else:
           raise ValueError("Response must be a string or a content object.")
      


   def __init__(self, response=None, **data):
       """
       Initialize the ActivityResponse.


       Args:
           response: The response content.
           **data: Additional data for initialization.


       Example:
           response1 = ActivityResponse("Hello, world!")
           response2 = ActivityResponse(TextContent(text="Hello, world!"))
       """
       if isinstance(response, str):
           super().__init__(response=TextContent(text=response), **data)
       else:
           super().__init__(response=response, **data)


   def to_dict(self):
       """
       Convert the ActivityResponse to a dictionary.


       Returns:
           dict: A dictionary representation of the response.


       Example:
           response = ActivityResponse("Hello, world!")
           assert response.to_dict() == {
               "response_type": "text/plain",
               "response": {"text": "Hello, world!"}
           }
       """
       return {
           "response_type": self.response_type.value,
           "response": self.response.to_dict()
       }


   def to_json(self):
       """
       Convert the ActivityResponse to a JSON string.


       Returns:
           str: A JSON string representation of the response.


       Example:
           response = ActivityResponse("Hello, world!")
           json_str = response.to_json()
           assert json.loads(json_str) == response.to_dict()
       """
       return json.dumps(self.to_dict())
```
