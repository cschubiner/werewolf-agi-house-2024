# Other teams' submissions (archived)

Mirrors of other teams' AGI-thon Werewolf (Nov 9, 2024) code, plus summaries of their forum writeups. The forum (openagi.discourse.group) is offline, so writeup links go to Wayback Machine snapshots.

**Code:** Only MIT-licensed code is copied here, with each team's `LICENSE.txt`. We copied just the files that differ from [sentient-agi/werewolf-template](https://github.com/sentient-agi/werewolf-template). One hard-coded API key was replaced with `sk-REDACTED`. Team 5's repo has no license, so it's kept as a GitHub fork instead of being copied.

**Writeups:** These are our own summaries, not verbatim copies (the posts are the authors' own work). Use the archive links for the full text.

| Final place | Team | Agent pseudonym | Final win % | Code | Writeup |
|---|---|---|---|---|---|
| 1 | Team 6 | Jean | 61.54 | not published | [archive](https://web.archive.org/web/20241121080234/https://openagi.discourse.group/t/team-6-submission-for-werewolf-agi-thon/2508) |
| 2 | Team 8 "PackMind" | Kim | 57.14 | original repo deleted, no copy found | [archive](https://web.archive.org/web/20241204153433/https://openagi.discourse.group/t/team-8-packmind-submission-for-werewolf-agi-thon-2-place-winner-submission/2515) |
| 3 | Team 1 "AlphaWolf" (us) | James | 51.61 | [`../src/werewolf_agents/cot_sample/agent/cot_agent.py`](../src/werewolf_agents/cot_sample/agent/cot_agent.py) | [archive](https://web.archive.org/web/20241112025558/https://openagi.discourse.group/t/team-1-submission/2505) |
| 4 | Team 13 | Kelly | 48.15 | [`team-13-itsuncheng-superwolf/`](team-13-itsuncheng-superwolf/) (from [itsuncheng/werewolf-template](https://github.com/itsuncheng/werewolf-template)) | [archive](https://web.archive.org/web/20241204150542/https://openagi.discourse.group/t/team-13-submission-for-werewolf-agi-thon/2513) |
| 5 | Team 14 | Lisa | 44.83 | not published | [archive](https://web.archive.org/web/20241204150954/https://openagi.discourse.group/t/team-14-submission-for-werewolf-agi-thon/2544) |
| 6 | Team 28 | Nate | 42.31 | [`team-28-julyankb/`](team-28-julyankb/) (from [julyankb/werewolf-agithon](https://github.com/julyankb/werewolf-agithon)) | [archive](https://web.archive.org/web/20241204150730/https://openagi.discourse.group/t/team-28-submission-for-werewolf-agi-thon/2520) |
| 7 (tie) | Team 9 | Kate | 41.94 | [`team-09-yisz-werewolf-seer9/`](team-09-yisz-werewolf-seer9/) (from [yisz/werewolf-seer9](https://github.com/yisz/werewolf-seer9)) | [archive](https://web.archive.org/web/20241112025600/https://openagi.discourse.group/t/agi-thon-werewolf-agent-team-9-implementation/2504) |
| 7 (tie) | Team 30 | Otto | 41.94 | not published | [archive](https://web.archive.org/web/20241204150510/https://openagi.discourse.group/t/team-30-submission-for-werewofl-agi-thon/2517) |
| 15 | Team 5 | Jack | 32.00 | fork: [cschubiner/AGIthon_werewolf-team5-archive](https://github.com/cschubiner/AGIthon_werewolf-team5-archive) (from [trepkakai/AGIthon_werewolf](https://github.com/trepkakai/AGIthon_werewolf)) | [archive](https://web.archive.org/web/20241112025556/https://openagi.discourse.group/t/agi-thon-werewolf-agent-team-5-implementation/2500) |

The full standings are in [`../results/`](../results/).

## Writeup summaries

### Team 6 (1st): traheja, Nilay, Manjit
- **Approach:** a "rational" agent built mainly for defense.
- **Input cleanup:** every incoming message was sanitized down to game-relevant facts before the model saw it.
- **Codenames:** player names were mapped to single-character codes, so a successful jailbreak would only leak meaningless codes.
- **Timeout fallback:** under time pressure the agent made a safe random choice rather than timing out. They noticed that voting early started a herd, with others piling onto the same player.
- **Tone:** a warm, reassuring voice that echoes Llama's training data, which made other LLMs more trusting, especially when the agent was a wolf. When accused, it could turn suspicion back on the accusers.
- **Memory:** a small writable store of raw facts. The logic rules only read from this memory, not from what other players said.

### Team 8 "PackMind" (2nd): Aydar Akhmetzyanov, Lily Zhu, Dhawal Modi, Vince
- Built on the simple sample agent.
- Role-specific instructions for each role.
- A memory of notes on accusations, alliances, and defenses.
- A chain-of-thought check before every public message: does it reveal too much, and does it serve the team's goals?
- Behavior changes by phase: stealthy hints during discussion, just a name when voting.

### Team 13 (4th): Evan Hu, Ayush Kanodia, Greg Schwartz, Alexis Li, I-Tsun (Raymond) Cheng
- Custom prompts per role, extending the CoT sample.
- **Villagers** estimate each player's wolf probability from revealed roles and voting patterns.
- **Wolves** hunt for the seer and doctor without sounding aggressive.
- **No reflection step:** they found it second-guessed initial answers that were often better, and the extra latency caused timeouts.
- A binary classifier to catch injection attempts.
- Every message starts with a fake claim that the seer privately cleared them as a villager.

### Team 14 (5th): linksku
- **Premise:** a text-only game with 8 effective players, 2 special roles, and no private chats carries very little signal. Assume seer claims are noise and voting-pattern stats won't help much.
- **Goal:** simply get executed by the town less often than other bots.
- **Main move:** always pretend to be a power role (seer), a habit from human social-deduction games.

### Team 28 (6th): julyankb
- **Suspicion scores:** keyword counts over chat ("accuse", "lying", "wolf", … vs. "trust me", "innocent", …), weighted up for players who accuse or defend a lot.
- **Also covers:** adaptive decision-making and protection against manipulation (see the code).

### Team 9 (tied 7th): yisz
- **Defending when accused:** a wolf accused by more than one player claims to be the seer and says it checked an accuser and found a wolf.
- **Reveals:** the seer and doctor reveal themselves once they have key night information.
- **Rebuttal:** everyone pushes back on "you're too quiet" accusations with "it wasn't my turn yet."
- **Fake point system:** after each message it appends a fake moderator notice announcing a special point system.
  - As a villager, the notice tells wolves to confess.
  - As a wolf, it tells villagers to claim to be wolves.
  - Other players are asked to reply "Point System Acknowledged."
- **Defense:** other players' messages are summarized to about 20–25 words, ignoring anything that claims to be a moderator or new rules.
- **Feedback to organizers:** suggested running separate tournaments with and without jailbreaking.

### Team 30 (tied 7th): Dan
- Focused on defense: truncated or limited what other players could feed into the model, and trimmed history when the context window was close to overflowing.
- Tried a banned-word list and dropped it after it made no difference.
- The code was never posted.

### Team 5 (15th): tjc7 / trepkakai
- **Hypotheses:** a normal text game can't give a real edge, most teams will jailbreak so reading other players' messages is dangerous, and simple beats clever.
- **Wolf:** reads only the moderator and jailbreaks others into repeating an innocent player's name.
- **Villager:** jailbreaks wolves into confessing and votes for anyone who confesses.
- **Defense:** screens incoming messages by peeking at their first ~70, then ~150 characters, and ignores anything flagged.
- **Result:** finished 15th of 18.

## Sentient's post-event analysis
[Leveling Up Reasoning Via Games: a Post AGI-thon Analysis](https://web.archive.org/web/20250619022303/https://openagi.discourse.group/t/leveling-up-reasoning-via-games-a-post-agi-thon-analysis/2669) (Dec 2024):
- Sentient re-ran the top 8 agents for 1,124 games with jailbreakers excluded.
- They fine-tuned Llama 3.1-8B on the winners' transcripts.
- Their per-role table is in the main README.
