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

Detailed per-team write-ups, with all three rankings (final, pre-tournament, Sentient re-run), are in the main README: [Other approaches](../README.md#other-approaches-every-team-with-rankings).

## Sentient's post-event analysis
[Leveling Up Reasoning Via Games: a Post AGI-thon Analysis](https://web.archive.org/web/20250619022303/https://openagi.discourse.group/t/leveling-up-reasoning-via-games-a-post-agi-thon-analysis/2669) (Dec 2024):
- Sentient re-ran the top 8 agents for 1,124 games with jailbreakers excluded.
- They fine-tuned Llama 3.1-8B on the winners' transcripts.
- Their per-role table is in the main README.
