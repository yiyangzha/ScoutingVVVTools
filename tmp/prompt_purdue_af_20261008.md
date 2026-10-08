# CMS Run 3 VVV scouting: continue the fresh campaign on the Purdue Analysis Facility

> **Status: superseded draft (2026-10-08T20:20-04:00).** After this draft was written, Yiyang decided to stay on FASRC Cannon and to drive the conversion of the IHEP-hosted inputs on IHEP over SSH (`tmp/fasrc_20261008/REQUIREMENTS.md` items 42-43). Kept at Yiyang's request as a reference for a possible later move to the Purdue AF; its "items 37-43" numbering predates the final REQUIREMENTS numbering.

Prepared on 2026-10-08 (America/New_York) by the Claude Code agent that ran the FASRC Cannon campaign `fasrc_20261008`. Part A is for Yiyang to execute by hand on the Purdue Analysis Facility (AF). Part B is the new agent's task. This handoff is in English. Every request for Yiyang's input must use a visible Plan-style question card in Chinese (Claude Code: `AskUserQuestion`), including confirmations, installation/download permission, deletion permission, inaccessible resources and problems. Ordinary messages only report progress.

Source of this handoff:

- Repository: `https://github.com/yiyangzha/ScoutingVVVTools` (public; `git@github.com:yiyangzha/ScoutingVVVTools.git`).
- Branch: `fasrc-campaign-20261008`, created from `main` at `fbe3bc40214ede6366d999605d5b06b361679179` (PR #10). It contains all code changes of the Cannon campaign, `pixi.lock`, this prompt, the original campaign prompt `tmp/prompt.md`, and the Cannon campaign records `tmp/fasrc_20261008/{REQUIREMENTS.md,PLAN.md,LOG.md,progress.json}`. Yiyang explicitly authorized this one commit and push; git stays read-only for the agent.
- Why the move: on Cannon the inputs are remote. The 2024G data and 32 MC samples are only at IHEP Beijing and read at 0.2-0.3 MB/s per stream (Part B5). Yiyang chose to continue on the Purdue AF with the **CERN account** (card answer, 2026-10-08), redoing every stage there and bringing no Cannon products.

Official AF documentation used below: https://analysis-facility.physics.purdue.edu/ (pages `getting-started/`, `login-methods/`, `storage/`, `data-access/`, `software/`, `gpus/`, `scaling-out/`, `guide-pixi/`, `guide-combine/`, `guide-dask-gateway/`, `guide-cern-eos/`, `guide-eos-write/`, `guide-file-transfer/`, `troubleshooting/`). Claude Code installation: https://code.claude.com/docs/en/setup.

## Part A — Yiyang's setup on the Purdue AF (CERN account)

Run the blocks in order and check each result. Running an installation/copy block approves that specific action only. Keep existing files; back up a settings file before replacing it. If something differs from the expectations, start the agent anyway and let it ask through a card.

### A0. What the CERN-account login gives you (from the AF docs)

- Login: https://cms.geddes.rcac.purdue.edu/hub through CILogon, choosing the CERN account. The AF username is the CERN username with the suffix `-cern` (expected `yiyangz-cern`); each login method has its own home, `/work` and quotas, so always use the CERN login.
- Available: JupyterLab sessions with chosen CPU/RAM/GPU (A100 full 40 GB "subject to availability", A100 5 GB slice, T4 16 GB), Dask Gateway (up to 1000 workers / 1000 cores / 6000 GiB per cluster, the first 100 cores guaranteed, at most 64 cores and 64 GiB per worker, idle clusters stop after 1 h), CVMFS, `/eos/purdue` read-only, CERNBox at `/eos/cern/home-y/yiyangz/` after `eos-connect` (Kerberos, renewed every session).
- Not available with the CERN login: Slurm (Hammer), writing to `/depot` (read-only).
- Storage: `/home/<user>` 25 GB strict (over quota: sessions cannot start; pixi refuses to run projects under `/home`), `/work/users/<user>/` 100 GB (not filesystem-enforced; admins contact users above it), `/work/projects/<name>/` up to 1 TB on request, `/tmp/<user>` session-local. No backup or purge policy is documented; the AF says to save often and sync code with git.
- Sessions: AF sessions are EL8 containers. Idle sessions are shut down automatically, idle GPU sessions sooner; a session also ends at its time limit (24 h per Yiyang). `/home` and `/work` survive the end of a session; running processes (the agent, conversions) and `/tmp` do not.

Recommended session shapes:

| Stage | GPU | CPUs | Memory |
| --- | --- | --- | --- |
| Setup, PU, conversion, mix, SR, QCD, systematics, plots, combine | none | 32 (more if offered: conversion scales with cores) | 128 GB |
| BDT training (5 trees) | full A100 40 GB | 16-32 | 128-192 GB |

Keep the A100 only for training days: CPU stages do not use it, the full A100 is scarce, and idle GPU sessions are culled sooner.

### A1. Open a session and check the environment

Start a CPU session (32 CPUs, 128 GB, no GPU), open a JupyterLab terminal and run:

```bash
whoami
echo "$HOME"
cat /etc/os-release | head -3
df -h "$HOME" /work/users/"$USER"
ls -d /cvmfs/cms.cern.ch/el8_amd64_gcc13/cms/cmssw/CMSSW_16_1_0_pre4
command -v pixi git curl xrdcp xrdfs voms-proxy-init voms-proxy-info gfal-copy dasgoclient tmux eos-connect || true
```

Expect `yiyangz-cern`, an EL8 system, and the el8 CMSSW release directory (it exists on CVMFS; seen from Cannon). Record which commands are missing; the agent handles them through cards.

### A2. Persistent shell settings

```bash
if grep -q -F '# >>> scouting-vvv (Purdue AF) >>>' "$HOME/.bashrc"; then
  sed -n '/^# >>> scouting-vvv (Purdue AF) >>>$/,/^# <<< scouting-vvv (Purdue AF) <<<$/p' "$HOME/.bashrc"
else
  cp -p --no-clobber "$HOME/.bashrc" "$HOME/.bashrc.before-vvv-$(date +%Y%m%dT%H%M%S)" 2>/dev/null || true
  cat >> "$HOME/.bashrc" <<'EOF'
# >>> scouting-vvv (Purdue AF) >>>
export VVV_PROJECT=/work/users/$USER/CMS_Run3_VVV_Scouting
export VVV_REPO=$VVV_PROJECT/ScoutingVVVTools
export PIXI_CACHE_DIR=/work/users/$USER/.cache/pixi
export PATH="$HOME/.local/bin:$PATH"
# <<< scouting-vvv (Purdue AF) <<<
EOF
fi
source "$HOME/.bashrc"
mkdir -p "$VVV_PROJECT" "$PIXI_CACHE_DIR"
```

The pixi cache goes to `/work` because `/home` is only 25 GB.

### A3. Clone the campaign branch

```bash
cd "$VVV_PROJECT"
git clone --branch fasrc-campaign-20261008 https://github.com/yiyangzha/ScoutingVVVTools.git ScoutingVVVTools
cd "$VVV_REPO"
git status --short
git log -1 --oneline
git rev-parse HEAD
```

The repository is public, so HTTPS needs no credentials. If `$VVV_REPO` already exists, inspect it instead of cloning over it.

### A4. Install Claude Code and copy its settings

The official native installer puts the launcher in `~/.local/bin/claude` and the binary in `~/.local/share/claude/` (a few hundred MB, fine for the 25 GB home):

```bash
command -v claude || curl -fsSL https://claude.ai/install.sh | bash
source "$HOME/.bashrc"
claude --version
mkdir -p "$HOME/.claude"
for f in settings.json CLAUDE.md; do
  [ -e "$HOME/.claude/$f" ] && cp -p --no-clobber "$HOME/.claude/$f" "$HOME/.claude/$f.before-vvv-$(date +%Y%m%dT%H%M%S)"
done
```

Global Claude settings (same as on Cannon):

```bash
cat > "$HOME/.claude/settings.json" <<'EOF'
{
  "env": {
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_FEEDBACK_COMMAND": "1"
  },
  "model": "opus",
  "modelSettings": {
    "claude-opus-5": {
      "effortLevel": "xhigh"
    },
    "claude-opus-5-5": {
      "effortLevel": "high"
    }
  },
  "theme": "auto"
}
EOF
```

Global instructions `~/.claude/CLAUDE.md` (identical to Cannon's `~/.claude/CLAUDE.md` and `~/.codex/AGENTS.md`; the copy on lxplus is `/afs/cern.ch/user/y/yiyangz/.codex/AGENTS.md`). Note that its first bullet ("only the current repository") is narrowed by the campaign write scope in Part B2:

~~~~bash
cat > "$HOME/.claude/CLAUDE.md" <<'EOF'
# AGENTS.md

- Do not modify, add, or delete any files outside the current repository directory. Any temporary or intermediate files must also be created inside the current repository directory.
- Do not install or download any files, tools, packages, fonts, dependencies, or external resources unless the user explicitly permits that exact action.
- Always address the root cause. Use the correct logic to directly resolve the issue. Don't just apply a quick fix. Avoid writing the negation of a negation. Instead, use affirmative descriptions.
- Run at most two subagents concurrently. Choose each subagent’s model and reasoning effort based on task difficulty, not necessarily matching the main agent; use `medium` or `high` as appropriate and `xhigh` only when strictly necessary.
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. Guardrails

- Never overwrite user edits between reads.
- Never restore deleted code without confirmation.
- Make the smallest fix that solves the problem.
- No scope drift: no refactors, restyles, or extras unless asked.
- Fix root causes, not symptoms.
- Use web search for unstable or version-specific behavior; cite sources.
- State assumptions; ask only when blocked.
- Briefly narrate multi-step tool usage.
- Finish the full plan once started.

## 1. Non-Destructive File Handling

**Preserve user data. Prefer additive changes over destructive ones.**

- Never delete files or directories.
- Never run destructive commands or cleanup operations that remove user data.
- Do not use `rm`, `del`, `erase`, `rmdir`, `Remove-Item`, or equivalents.
- Do not remove files just because they look temporary, generated, redundant, cached, old, or replaceable.
- Do not delete files before recreating or renaming them.
- If replacement is needed, write a new file and leave the original intact.
- If renaming fails, keep the old file and create a new one with a different name.
- If deletion seems necessary, stop and propose a non-destructive alternative first.

Default rule: when in doubt, keep all existing files.

## 2. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 3. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 4. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 5. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
EOF
chmod 600 "$HOME/.claude/settings.json" "$HOME/.claude/CLAUDE.md"
~~~~

Then run `claude` once in the terminal and complete the browser login it prints. Do not copy `~/.claude/.credentials.json`, session histories or plugins from another machine. Codex is not needed for this handoff.

### A5. Grid certificate and VOMS proxy

The AF docs require uploading `usercert.pem` and `userkey.pem` and creating the proxy with `voms-proxy-init --rfc --voms cms -valid 192:00`. Dask workers only see a proxy stored under `/work/users/$USER/`, so create it there. Copy the PEM files from lxplus with an outgoing scp from the AF terminal (or upload them through the JupyterLab file browser):

```bash
umask 077
mkdir -p "$HOME/.globus"
chmod 700 "$HOME/.globus"
scp yiyangz@lxplus.cern.ch:.globus/usercert.pem yiyangz@lxplus.cern.ch:.globus/userkey.pem "$HOME/.globus/"
chmod 400 "$HOME/.globus/usercert.pem" "$HOME/.globus/userkey.pem"
openssl x509 -in "$HOME/.globus/usercert.pem" -noout -subject -dates
```

If the lxplus copies are elsewhere, use the actual paths (Cannon has them in `~/.globus` too). Create the proxy (enter the key passphrase):

```bash
umask 077
unset X509_USER_PROXY
if mkdir -p /work/users/$USER/proxies && chmod 700 /work/users/$USER/proxies &&
   CMS_PROXY_DIR=$(mktemp -d /work/users/$USER/proxies/cms-XXXXXX); then
  if voms-proxy-init --rfc --voms cms --valid 192:00 \
       --cert "$HOME/.globus/usercert.pem" --key "$HOME/.globus/userkey.pem" \
       --out "$CMS_PROXY_DIR/x509up"; then
    export X509_USER_PROXY="$CMS_PROXY_DIR/x509up"
    chmod 600 "$X509_USER_PROXY"
    voms-proxy-info --file "$X509_USER_PROXY" --vo --timeleft --actimeleft
  else
    echo "Proxy creation failed; keep the output for the agent." >&2
  fi
fi
```

Do not export `X509_USER_CERT`/`X509_USER_KEY` in the agent's shell: on Cannon jobs that inherited them made the XRootD TLS client try the passphrase-protected key and fail (Part B5).

### A6. Copy the PU reference histograms and historical records from CERNBox

The three official 2024 C-I pileup reference histograms (authorized external inputs, `*.root` is git-ignored) and the lxplus historical records are in the old lxplus checkout on your CERNBox:

```bash
eos-connect            # CERN username/password; needed again in every new session
SRC=/eos/cern/home-y/yiyangz/CMS_Run3_VVV_Scouting/ScoutingVVVTools
ls -l "$SRC/tmp/lxplus_full_20261001/inputs/pileup/"
cd "$VVV_REPO"
mkdir -p tmp/lxplus_full_20261001/inputs/pileup selections/weight/pileup
cp -p --no-clobber "$SRC"/tmp/lxplus_full_20261001/inputs/pileup/dataPileupHistogram-2024CDEFGHI_Golden-*.root tmp/lxplus_full_20261001/inputs/pileup/
cp -p --no-clobber "$SRC"/tmp/lxplus_full_20261001/inputs/pileup/dataPileupHistogram-2024CDEFGHI_Golden-*.root selections/weight/pileup/
cp -p --no-clobber "$SRC"/tmp/prompt_lxplus_20261001_backup.md "$SRC"/tmp/lxplus_workflow_20261001_plan.md \
   "$SRC"/tmp/lxplus_workflow_20261001_log.md "$SRC"/tmp/lxplus_workflow_20261001_progress.json tmp/
sha256sum selections/weight/pileup/dataPileupHistogram-2024CDEFGHI_Golden-*.root
```

Expected SHA256: `66000ub` 02be96ef021370cb8c6e6d66e49bc40bf3e9e9c75711b7d7491c70d7c2f97454, `69200ub` 5e379a2004788f693187f6a579a81f2530a27a00482e39b559337d026c40fb93, `72400ub` 3b85a78a4a1ee929edccf9b785e5c50095022cdd74e5a29539fba4842dfcb253. If CERNBox access fails, use `scp yiyangz@lxplus.cern.ch:/eos/home-y/yiyangz/CMS_Run3_VVV_Scouting/ScoutingVVVTools/tmp/...` from the AF terminal instead.

### A7. Session environment and start of the agent

```bash
cd "$VVV_REPO"
mkdir -p build/purdue_af
cat > build/purdue_af/session.env <<EOF
source "\$HOME/.bashrc"
export X509_USER_PROXY="${X509_USER_PROXY:-}"
unset X509_USER_CERT X509_USER_KEY
EOF
chmod 600 build/purdue_af/session.env
command -v tmux && tmux new -s vvv   # optional; JupyterLab terminals also keep running when the browser closes
source build/purdue_af/session.env
cd "$VVV_REPO"
claude
```

Give the new agent this startup instruction:

> Read tmp/prompt_purdue_af_20261008.md completely and execute Part B. Read the current AGENTS.md, README.md, tmp/prompt.md and the records in tmp/fasrc_20261008/ completely. Continue the fresh campaign on the Purdue AF with my CERN account, redoing every stage from the registered source MC and the 2024G_UParT_v6 data. Every matter requiring my input must be a visible Plan-style question card in Chinese, including confirmations and download/deletion permission; do not ask in ordinary messages. Verify the question-card feature at the first decision. You may adapt the server integration; ask before any physics or analysis-logic change. Keep working, including periodic monitoring, until the task is complete, a question awaits my reply, or an unsolvable problem blocks progress.

### A8. Every new session (after the 24 h limit or a culled session)

1. Start a session with the shape of the current stage (A0 table).
2. `eos-connect` if CERNBox is needed; check the proxy with `voms-proxy-info --file "$X509_USER_PROXY" --timeleft`, renew it with the A5 block (new directory, then update `build/purdue_af/session.env`) when it is short.
3. `source /work/users/$USER/CMS_Run3_VVV_Scouting/ScoutingVVVTools/build/purdue_af/session.env; cd "$VVV_REPO"; claude --continue`, then tell the agent "继续" (it rereads its records and restarts the interrupted stage; completed conversion batches are kept by `resume_successful_batches`).

## Part B — instructions for the new agent

### B1. Objective and scope (unchanged)

Complete the fresh analysis campaign: PU weights → ttbar control region conversion → JMS/JMR fit → analysis conversion (five trees `boosted3`, `boosted2_resolved`, `boosted2_unresolved`, `boosted2_leptonic_1lep`, `boosted2_leptonic_2lep`, plus the eight MC jet variations per tree) → mix → BDT training (GPU) → signal regions → QCD ABCD → all evaluable systematics → Data/MC and class-shape plots → combine expected significance and limits (combined, signal-class, signal-sample, and per-channel scenarios; MC-true and ABCD QCD) → agreed standalone studies (scope by card). Inputs: the registered MC in `src/sample.json` within the approved scope (BDT `class_groups` ∪ ttbar-CR submit list, 83 MC samples) and only the data `2024G_UParT_v6`. Expected sensitivity at 255.8 fb^-1; keep the 2024 C-I PU profile. Every stage is redone on the AF; no Cannon product is reused. Completion means real outputs for every stage, validated, not submissions or zero exit codes.

The full binding text of the original handoff (`tmp/prompt.md` Part B, sections B1-B8) and every later decision are transcribed in `tmp/fasrc_20261008/REQUIREMENTS.md` (items 1-36) plus B3 below (items 37-43). Where Cannon-specific details there (paths, Slurm, Cannon write scope, proxy backup) conflict with this prompt, this prompt wins; the physics decisions stay.

### B2. Interaction, safety and persistence rules

1. Every request for input (clarification, confirmation, plan approval, installation/download, deletion, conflicting instructions, inaccessible resources, physics changes) is a visible Chinese `AskUserQuestion` card with evidence, concrete options, a recommendation and its reason; one round at a time; wait for the answer, restate it, record it. Ordinary messages report progress only. Verify the card at the first decision.
2. Keep executing until the objective is achieved; stop only while a card awaits an answer or for an unsolvable blocker. Monitor running work periodically (about 2-5 minutes, bounded waits), report short Chinese progress messages, and keep timestamped records.
3. Preserve every file. No deletion of anything (intermediates, caches, failed outputs, program-driven cleanup) without a prior explicit card approval; collect necessary deletion proposals for one final decision. Already approved program-internal cleanups: the three `convert_branch` cleanups (per-thread temp ROOT files after their verified fast merge, the hidden partial output of a failed merge, a batch's old `.meta` before reprocessing). `combine` `keep_work` stays true. Audit other cleanup before running programs.
4. Write scope: the repository `/work/users/$USER/CMS_Run3_VVV_Scouting/ScoutingVVVTools`, the pixi cache `/work/users/$USER/.cache/pixi`, the proxy directory `/work/users/$USER/proxies` (Part A), and session `/tmp/$USER`. Intermediate and ntuple locations beyond the repository need a card (B6). `/eos/cern`, `/eos/purdue`, `/depot` and other users' areas are read-only unless a card approves a write.
5. Git is read-only (status, diff, log, show, check-ignore). Yiyang explicitly authorized only the one commit/push that created this branch. Ask before any `.gitignore` change; report the tracked/ignored status of new files at the end.
6. No installations or downloads without a card naming source, version, destination, command and purpose. The repository pixi environments (`default` CPU, `gpu` CUDA; `pixi.lock` committed) may be installed from the lock file (approved on Cannon; same manifest). Remote DAS/XRootD reads are approved for the MC in scope and `2024G_UParT_v6` only.
7. Execution: the AF session is the allocated compute container (there is no login/worker split and no Slurm with the CERN login). Normal campaign compilation and execution in the session (and on Dask Gateway workers once validated) are authorized as on Cannon; ad-hoc tests beyond the campaign pilots, or anything AGENTS.md's testing rules leave unclear, go through a card. Pilot every stage on a small subset and check the result before the full run.
8. Credentials: never print or copy key/proxy/auth contents; use the explicit proxy path; check `--timeleft` and `--actimeleft` before long runs; ask for renewal by card when short.
9. At most two subagents at a time, current model, effort by difficulty. Work slowly and carefully. Fix root causes; report unrelated problems instead of fixing them. Physics or analysis-logic changes need a card before editing.
10. Records: create `tmp/purdue_af_<YYYYMMDD>/{REQUIREMENTS.md,PLAN.md,LOG.md,progress.json}` in English (untracked), carry over requirements 1-43 verbatim or by exact reference, record commands, versions, hashes, provenance and answers with real timestamps. Keep `tmp/fasrc_20261008/` unchanged as history. At every small-stage checkpoint reread this prompt, AGENTS.md, README.md and the complete records word for word. Keep AGENTS.md (section 6 requirements) and README.md in sync with code changes. Conversation in Chinese; code comments and docs in English.

### B3. Confirmed decisions (summary; full text in `tmp/fasrc_20261008/REQUIREMENTS.md`)

Carried over from Cannon (items 13-36 there):

- Step results go to each step's default output directory (overwriting the tracked historical `selections/jms_jmr/output/jms_jmr_results.json` and `selections/pileup_syst/output/pileup_syst_yields.json` is approved); generated products stay untracked.
- AK4 resolved cleaning against the two leading AK8 jets (implemented: `size(ak8_kept2) >= 2 && deltaR(self, ak8_kept2[0]) > 0.8 && deltaR(self, ak8_kept2[1]) > 0.8`; `ak8_kept2` kept because outputs use it).
- Mix, then train on the `{sample_group}_mixed` files; variation trees are only read by the jet-syst scripts from the unmixed files.
- MC weights (physics): w = L·σ·genWeight·weight_pu / S × N_tree / N_read with S = `sum_genweight_pu` written by the convert merge; PU variations normalized by their own sums; jet-variation trees carry genWeight; theory ratios stay absolute; training weights |w|·Σw/Σ|w| per sample and split; every physics output uses signed weights. Shared code `selections/mc_weight_common/mc_weights.py`.
- Latest MC consistent with the main analysis; the ttbar-CR VV class uses the eight exclusive VV samples (old `ww`/`wz`/`zz` unreachable).
- SR optimizer: `signal_region_hist.py` is the single optimizer (mode 3), objective Cowan Z_A with background variance V = Σ_k(Σ_c δ_kc B_c)² + Σw² (each nuisance fully correlated across classes), event-mask deduplication, boundaries inside empty score gaps (signal-score axes as high as possible, background-score axes as low as possible), exact N regions, B&B time limit 14400 s; precision first, runtime optimized without losing precision.
- All prediction-reference checks restored (signal_region, qcd_est strict, data_mc score check). If GPU-train vs CPU-inference probability differences exceed the stored tolerances, report the measured values and ask.
- Data/MC display projected from 27.51 fb^-1 to 255.8 fb^-1 (data bins and errors scaled, labelled) — **approved but not implemented yet**; the final sensitivity uses 255.8 fb^-1.
- Systematic ratios use the full converted samples (training events included).

New on 2026-10-08 (items 37-43):

37. Data access on Cannon was too slow; the campaign moves to the Purdue AF; all stages are redone there; no Cannon products are transferred (the Purdue unreadable-file scan on Cannon was cancelled at Yiyang's request).
38. AF login: the CERN account (Yiyang's card answer); Slurm/depot are unavailable with it.
39. The campaign code, `pixi.lock`, the handoff prompts and the `fasrc_20261008` records were committed to the new branch `fasrc-campaign-20261008` and pushed (one-time explicit authorization; git otherwise read-only).
40. Long-term location of reusable ntuples on the AF: not decided; Yiyang asked for the options first. Ask by card after the pilot has measured the output size (B6).
41. Session shapes as in Part A0 (CPU stages 32 CPUs/128 GB without GPU; training with a full A100). A session lasts at most 24 h; work must be resumable across sessions.
42. If the AF turns out unworkable, the fallbacks discussed are: back to Cannon with IHEP reads relayed through the Purdue XCache (`root://xcache.cms.rcac.purdue.edu//`, measured 7x faster per stream than direct on a first read) or multi-stream staging; lxplus (HTCondor, CERNBox; measure the IHEP→CERN rate first); subMIT or CI Connect would need a full new setup; lxlogin (IHEP) is closest to the data but cannot run Claude. Any switch is Yiyang's decision by card.
43. Pending decision from Cannon: `weight.C` aborts a sample when one input file cannot be opened (the converter skips and records unreadable MC files). Re-test the unreadable files (B5) locally first; if some stay unreadable, ask whether to skip-and-record in `weight.C` or to wait for a fix of the files.

### B4. State of the code on this branch

Implemented on Cannon (see `tmp/fasrc_20261008/LOG.md` for every detail and README/AGENTS for the documentation):

- `run.py`: site profiles `--site {purdue,fasrc}` (`SITES`), `--x509-proxy`, `--python-env`; the fasrc profile runs C++ through `cmssw-el9` + `sites/fasrc/cmssw_exec.sh`, Python through `pixi run --frozen -e <env>`, combine through the CVMFS `combine-standalone` image, keeps content-hashed binaries, submits itself as a driver job, and removes `X509_USER_CERT`/`X509_USER_KEY` from job environments. Mode 3 runs `signal_region_hist.py` (`SR_HIST_CONFIG_PATH`). The default `purdue` profile is the old Hammer behavior and is not usable with the CERN login.
- `pixi.toml`/`pixi.lock`: `default` = CPU (`py-xgboost-cpu` 2.1.4) and `gpu` = CUDA (`py-xgboost-gpu` 2.1.4 cuda128, `cuda = "12"` system requirement); both with scikit-learn, scipy, zstandard (python 3.11, uproot 5.7.7). Verified on Cannon CPU and A100 workers (real `cuda:0` training of a synthetic model).
- `convert_branch.C`: signed sums `sum_genweight`, `sum_genweight_pu`, `sum_genweight_pu_up/_down` per batch `.meta`, written to `sample.json` at the merge (error if `sum_genweight_pu <= 0`); genWeight read before the preselection. Compiled and run successfully on Cannon in a CR data pilot (CMSSW_16_1_0_pre4 el9_amd64_gcc13, GCC 13.4, ROOT 6.36.11, correctionlib 2.7.0).
- `weight.C`: remote opens serialized by a mutex. Compiled and run on Cannon (PU pilot `wwz` validated; 62 other samples produced on Cannon but not transferred).
- Weight builders on the shared helper: `train.py` (also thread count from CPU affinity, CUDA logging), `signal_region.py`, `signal_region_hist.py` (rewritten: systematics in the objective, dedup, exact boundaries, OpenMP selector with a Python fallback), `qcd_est.py` (TTree metadata via `mktree`, strict checks), `data_mc.py`, `class_shapes.py`, `jms_jmr_fit.py`, `theory_syst.py`, `pileup_syst.py`, `jet_variation_common.py` + the four wrappers; configs switched to `event_reweight_branches = ["genWeight", "weight_pu"]`.
- `selections/signal_region/config_hist_<tree>.json` for the five trees (lumi 255.8, N = 4, outputs `output/<tree>_hist/`, `background_syst_jsons` pointing to `../<syst>/output/inclusive/*_syst_yields.json`).
- ttbar-CR configs use the eight exclusive VV samples.
- Not executed anywhere yet: the Python weight changes, `signal_region_hist.py`, `qcd_est.py`, the syst scripts, combine with the new inputs. Two independent reviews found no bugs after their fixes (LOG).
- Not yet done: the inclusive syst configs (`config_*inclusive*.json`, no `signal_region_csv`, output `output/inclusive/`) needed by the SR objective; the 255.8 fb^-1 Data/MC projection; campaign configs for the AF paths. The untracked Cannon campaign configs `selections/convert*/config_fasrc*.json` stayed on Cannon (not committed).

### B5. Evidence from Cannon to reuse

- Input volume (DAS, in scope): 14.47 TB. Data `2024G_UParT_v6`: 8 datasets, 66,228 files, 6.26 TB, 5,956,056,483 events, `/store/user/yiyangz/ScoutingNanov3/...`, hosted only at IHEP Beijing (no Rucio site record). 32 MC samples owned by yiyangz (1.76 TB) are also at IHEP; 52 MC samples owned by jschulte (6.45 TB, largest `ttbar_semilep`/`ttbar_had` 0.86 TB each) are at Purdue (`af-a00.cms.rcac.purdue.edu:1098`). The IHEP files are also directly at `root://cceos.ihep.ac.cn//eos/ihep/cms/<LFN>` (same speed as the redirector from Cannon).
- Read rates from Cannon: IHEP single stream 0.2-0.27 MB/s; `xrdcp --streams 8` 0.6-0.9 MB/s per file; 64 parallel multi-stream copies about 24 MB/s aggregate; ROOT reads in the converter do not profit from substreams. Purdue files 1.5-1.7 MB/s per stream. Through the Purdue XCache an IHEP file read at 1.5 MB/s on the first read and 1.8 MB/s cached (from Cannon); a second multi-stream test gave 0.4 MB/s. DAS queries are fast (8882 files in 2.5 s).
- CR data pilot (5 data files, 4 threads): 6:55 wall, 1 % CPU (I/O bound, ~83 s per file), 475,498 lumi-masked entries, output 422 kB; batch/merge/sidecars/processed-lumi JSON correct.
- Source metadata (one file per sample): genEventCount equals Events entries (no upstream filter); PSWeight title "[0] isr.murfac=2.0; [1] fsr.murfac=2.0; [2] isr.murfac=0.5; [3] fsr.murfac=0.5" (matches `theory_syst.py`); LHEPdfWeight 101 or 103 members; negative-genWeight fractions VVV 4-6 %, qq→VH 3 %, ttbar 0.4 %, VV 19-22 %, others 0.
- A full `xrdfs stat` of all 20,539 in-scope MC files succeeded, but opening these Purdue files failed on Cannon with `[3014] Unable to open file ... Network is unreachable (source)` (also through FNAL, the direct Purdue server, and the XCache with `[3011] Item not found (source)`). They are the first failing file of each PU job; others may follow. Re-test them from the AF (`root://eos.cms.rcac.purdue.edu//<LFN>` and `/eos/purdue/<LFN>`):

| Sample | LFN |
| --- | --- |
| www | /store/user/jschulte/ScoutingNano/WWW-4F_TuneCP5_13p6TeV_amcatnlo-pythia8/ScoutingNano_MC_WWW_v5_jetMatchFix/260715_124121/0000/scouting_nano_MC_22.root |
| GluGluZH-Zto2L-Hto2Wto2L2Nu | /store/user/jschulte/ScoutingNano/GluGluZH-Zto2L-Hto2Wto2L2Nu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-jhugen-pythia8/ScoutingNano_MC_GluGluZH-Zto2L-Hto2Wto2L2Nu_v5/260713_145042/0000/scouting_nano_MC_16.root |
| WminusH-WtoLNu-Hto2WtoLNu2Q | /store/user/jschulte/ScoutingNano/WminusH-WtoLNu-Hto2WtoLNu2Q_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-jhugen-pythia8/ScoutingNano_MC_WminusH-WtoLNu-Hto2WtoLNu2Q_v5/260713_144508/0000/scouting_nano_MC_21.root |
| WplusH-Wto2Q-Hto2Wto4Q | /store/user/jschulte/ScoutingNano/WplusH-Wto2Q-Hto2Wto4Q_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-jhugen-pythia8/ScoutingNano_MC_WplusH-Wto2Q-Hto2Wto4Q_v5/260715_122533/0000/scouting_nano_MC_5.root |
| WplusH-WtoLNu-Hto2WtoLNu2Q | /store/user/jschulte/ScoutingNano/WplusH-WtoLNu-Hto2WtoLNu2Q_M-125_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-jhugen-pythia8/ScoutingNano_MC_WplusH-WtoLNu-Hto2WtoLNu2Q_v5/260713_144721/0000/scouting_nano_MC_13.root |
| ttbar_had | /store/user/jschulte/ScoutingNano/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/ScoutingNano_MC_TTTo4Q_v5/260713_143222/0000/scouting_nano_MC_106.root |
| ttbar_semilep | /store/user/jschulte/ScoutingNano/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/ScoutingNano_MC_TTToLNu2Q_v5/260713_143253/0000/scouting_nano_MC_235.root |
| wjets_h100to400 | /store/user/jschulte/ScoutingNano/Wto2Q-3Jets_Bin-HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/ScoutingNano_MC_Wto2Q_HT-100to400_v5/260713_145411/0000/scouting_nano_MC_282.root |
| wjets_h400to800 | /store/user/jschulte/ScoutingNano/Wto2Q-3Jets_Bin-HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/ScoutingNano_MC_Wto2Q_HT-400to800_v5/260713_145603/0000/scouting_nano_MC_217.root |
| wjets_h800to1500 | /store/user/jschulte/ScoutingNano/Wto2Q-3Jets_Bin-HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/ScoutingNano_MC_Wto2Q_HT-800to1500_v5/260713_145646/0000/scouting_nano_MC_99.root |
| wjets_h1500to2500 | /store/user/jschulte/ScoutingNano/Wto2Q-3Jets_Bin-HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/ScoutingNano_MC_Wto2Q_HT-1500to2500_v5/260713_145445/0000/scouting_nano_MC_72.root |
| wjets_h2500 | /store/user/jschulte/ScoutingNano/Wto2Q-3Jets_Bin-HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/ScoutingNano_MC_Wto2Q_HT-2500_v5/260713_145526/0000/scouting_nano_MC_249.root |
| zjets_h100to400 | /store/user/jschulte/ScoutingNano/Zto2Q-4Jets_Bin-HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/ScoutingNano_MC_Zto2Q_HT-100to400_v5/260713_145725/0000/scouting_nano_MC_24.root |
| zjets_h400to800 | /store/user/jschulte/ScoutingNano/Zto2Q-4Jets_Bin-HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/ScoutingNano_MC_Zto2Q_HT-400to800_v5/260713_145945/0000/scouting_nano_MC_430.root |
| qcd_ht200to400 | /store/user/jschulte/ScoutingNano/QCD-4Jets_Bin-HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/ScoutingNano_MC_QCD_HT-200to400_v5/260713_142725/0000/scouting_nano_MC_284.root |
| qcd_ht400to600 | /store/user/jschulte/ScoutingNano/QCD-4Jets_Bin-HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/ScoutingNano_MC_QCD_HT-400to600_v5/260713_142803/0000/scouting_nano_MC_48.root |
| qcd_ht800to1000 | /store/user/jschulte/ScoutingNano/QCD-4Jets_Bin-HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8/ScoutingNano_MC_QCD_HT-800to1000_v5/260713_143036/0000/scouting_nano_MC_254.root |
| qcd_ht1200to1500 | /store/user/jschulte/ScoutingNano/QCD-4Jets_Bin-HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/ScoutingNano_MC_QCD_HT-1200to1500_v5/260713_142518/0000/scouting_nano_MC_120.root |
| qcd_ht1500to2000 | /store/user/jschulte/ScoutingNano/QCD-4Jets_Bin-HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8/ScoutingNano_MC_QCD_HT-1500to2000_v5_jetMatchFix/260731_184330/0000/scouting_nano_MC_267.root |
| qcd_ht2000 | /store/user/jschulte/ScoutingNano/QCD-4Jets_Bin-HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8/ScoutingNano_MC_QCD_HT-2000_v5/260713_142642/0000/scouting_nano_MC_305.root |

### B6. AF server integration to design (state the plan first; ask where noted)

Read the complete current sources and data flow before designing (AGENTS.md "Read first"), then present the concrete plan (files, functions, behavior, config keys, outputs, downstream, docs). Keep the `purdue` and `fasrc` profiles working.

1. **Site profile.** Add an AF profile (for example `--site purdue_af`) in `run.py`'s `SITES`, modular like `fasrc`. With the CERN login there is no Slurm: run the stages inside the session with bounded concurrency (existing local paths, `MAX_CONCURRENT_JOBS`, OpenMP threads matched to the session CPUs), resumable across session ends (mode 0 keeps valid batches; record per-stage state). Dask Gateway (up to 1000 cores) is an option for conversion only after validating that its workers mount `/work`, see CVMFS, run the converter binary and receive the proxy; otherwise stay in the session.
2. **C++ environment.** CVMFS has CMSSW_16_1_0_pre4 for `el8_amd64_gcc13` (and el9). Prefer a native `cmsenv` in a repository project area built with `SCRAM_ARCH=el8_amd64_gcc13` (git-ignored `CMSSW_*`) on the EL8 session; check GCC/ROOT/correctionlib versions against the el9 build used on Cannon (GCC 13.4, ROOT 6.36.11, correctionlib 2.7.0) and record them. The AF warns that Apptainer inside a session "is not guaranteed to always work"; use `cmssw-el9` only if the native build is impossible. A different compiler/ROOT environment for the converter is an environment change: card first.
3. **Python.** Install the repository pixi `default` env from `pixi.lock` (`pixi install --frozen`); install the `gpu` env in an A100 session and verify real `cuda:0` training. Set `TMPDIR`, `MPLCONFIGDIR`, `XDG_CACHE_HOME` to `/work` or `/tmp/$USER` (not `/home`).
4. **combine.** The Cannon plan used the CVMFS `combine-standalone` image (CombinedLimit v11.1.0) through singularity, which may not work inside an AF session. The AF documents conda-forge `cms-combine` (pixi; v11.1.0 matches the image) and a global env `/work/pixi/global`. Choosing either is an environment decision: card with the exact proposal; check `combine.C` (tested with v10.0.2) against it.
5. **Input access.** `convert_branch.C` (input discovery) and `weight.C` prefix every DAS LFN with `root://cms-xrd-global.cern.ch/`. On the AF, Purdue-hosted files are local (`root://eos.cms.rcac.purdue.edu//` or the read-only `/eos/purdue/` mount) and the AF recommends `root://xcache.cms.rcac.purdue.edu//` for remote files read more than once. Make the input prefix a site/config choice (a server adaptation; keep the `.files` snapshot and `.meta` input-slice hash semantics, which hash the full URLs), measure the IHEP data rate from the AF through the XCache and the global redirector with a pilot, and report the expected data-conversion time before the full run. The ttbar CR and the main conversion both read the full data: consider whether a single staged copy is cheaper, but any bulk copy location needs a card (storage).
6. **Proxy.** Explicit proxy under `/work/users/$USER/proxies/` (Dask-visible); no `X509_USER_CERT`/`X509_USER_KEY` in job environments.
7. **Storage (card).** Measure the per-file output size in the pilots, extrapolate ntuples (nominal + 8 variations for MC, mixed copies, CR) and intermediates, compare with `/work/users/$USER` (100 GB), and ask by card where to keep intermediates and reusable ntuples: `/work/projects/<name>` (up to 1 TB, request to AF support), CERNBox `/eos/cern/home-y/yiyangz/` (quota and speed to check), Purdue EOS `/store/user/yiyangz/` via `root://eos.cms.rcac.purdue.edu` (the AF docs disagree on whether CERN logins may write there; test with `xrdfs ... ls` first), or a later copy back to Cannon (`/n/holystore01/LABS/iaifi_lab/Lab/yiyangz/CMS_Run3_VVV_Scouting/ntuples/`, about 0.95 TB free for the whole lab on 2026-10-08).
8. **Campaign configs.** Create new untracked AF configs next to the checked-in ones (the converter needs the sibling `branch.json`/`selection.json`), with absolute cross-stage paths, the default step output directories, and a fresh namespace (`purdue_af_<YYYYMMDD>`). Downstream Python scripts resolve some relative paths against their script directory; check each.

### B7. Execution plan and acceptance (same stages as `tmp/prompt.md` B7, adapted)

1. Inventory (session OS/CPU/RAM/GPU, quotas, tools, proxy VO/lifetimes, CERNBox, CVMFS releases), records, first card. Accept: working card, recorded environment.
2. Server integration (B6) with a stated plan; environment builds; pilot of each tool on one small input. Accept: native or approved environment builds the three C++ tools; pixi CPU/GPU envs verified.
3. Re-test the unreadable Purdue files (B5) and measure IHEP/Purdue read rates from the AF. Accept: explicit readable/unreadable list and a time estimate; cards for blockers (item 43).
4. PU (all 83 MC; pilot one sample first), ttbar CR (MC + data; pilot first), JMS/JMR fit with the actual 27.51 fb^-1 CR statistics. Accept: complete CR coverage, valid fit and diagnostics.
5. Analysis conversion with nominal corrections, new PU/JMS/JMR, eight MC variations; pilots of a signal, a QCD, a ttbar and a data batch (time, size, entries, `sum_genweight*` vs Runs `genEventSumw`); full production resumable across sessions; validate every batch/merge/sidecar. Accept: full coverage, consistent bookkeeping, storage decided by card.
6. Mix, then GPU training of five trees (A100 session). Accept: models, diagnostics, reference checks pass (or measured differences reported by card).
7. Inclusive systematics (modes 8/9/11-14 with inclusive configs), then SR optimization per tree, QCD ABCD, per-SR systematics, Data/MC (with the approved 255.8 fb^-1 projection implemented first) and class-shape plots. Accept: all nuisance inputs present, no silent unity.
8. combine (environment by card): all scenarios, per channel, MC-true and ABCD; inspect cards, fits and quantiles. Standalone studies by card.
9. Final verification: reread all records; report outputs, provenance, limitations; one final card for any deletion proposals (including the Cannon leftovers below) and the ntuple copy; git status / check-ignore of new files.

### B8. Cannon leftovers (not deleted; for the final deletion card only)

On Cannon (`/n/holystore01/LABS/iaifi_lab/Lab/yiyangz/CMS_Run3_VVV_Scouting/ScoutingVVVTools` and `/n/netscratch/iaifi_lab/Lab/yiyangz/CMS_Run3_VVV_Scouting/`): pilot outputs under `fasrc_20261008/pilot/`, the pixi cache and tmp on scratch, the repository `.pixi/` environments, `CMSSW_16_1_0_pre4/`, built binaries under `build/fasrc/bin/` and `selections/weight/weight_50143cc8e655`, 63 fresh PU CSV/PDF files in `selections/weight/pileup/`, PU job logs in `selections/weight/`, diagnostics under `build/fasrc/fasrc_20261008/`, the untracked `config_fasrc*.json`, and the private proxy backup `build/fasrc/credentials/cms-20261008T101031-0400/` (expires about 2026-10-16 09:52 EDT). Nothing on Cannon is running.
