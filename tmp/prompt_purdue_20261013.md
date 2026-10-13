# CMS Run 3 VVV scouting: continue the campaign at Purdue (Hammer, Purdue guest account)

Prepared on 2026-10-13 (America/New_York) by the Claude Code agent on FASRC Cannon. It supersedes `tmp/prompt_purdue_af_20261008.md` (an Analysis Facility draft for the CERN login, kept as history). Part A is for Yiyang to run by hand at Purdue. Part B is the task of the new agent. Every request for Yiyang's input is a visible Chinese Plan-style question card (Claude Code: `AskUserQuestion`); ordinary messages only report progress. Conversation in Chinese; code, comments, documentation and records in English.

Source of this handoff:

- Repository `https://github.com/yiyangzha/ScoutingVVVTools` (public), branch `fasrc-campaign-20261008`. It holds the Cannon campaign code, the merge of Jan's branch `JanFSchulte/ScoutingVVVTools:optimizations` (tip e972efa, his sensitivity review of 2026-10-13: `selections/BDT/study_boosted3/significance_review.md`), the MC-statistics/systematics-aware SR optimizer, this prompt and the records `tmp/fasrc_20261008/{REQUIREMENTS.md,PLAN.md,LOG.md,progress.json}` (requirements 1-66).
- Why Purdue: on Cannon all inputs are remote; Yiyang moves the whole work to Purdue with a Purdue guest (career) account, which gives Slurm on Hammer and writable `/depot/cms`. The IHEP work (`--site ihep`, `remote_input_rules`) stays in the code but is no longer used. The setup follows Jan's working Purdue setup: CMSSW environment with `cmsenv`, CMSSW's Python, Slurm on Hammer.

Official documentation used: Purdue CMS Tier-2 accounts https://www.physics.purdue.edu/Tier2/user-info/accounts/get-account.php, login https://www.physics.purdue.edu/Tier2/user-info/login.php, jobs https://www.physics.purdue.edu/Tier2/user-info/Submitting%20Jobs.php, Analysis Facility https://analysis-facility.physics.purdue.edu/ (login-methods, storage, scaling-out, getting-started), RCAC Hammer storage https://www.rcac.purdue.edu/knowledge/hammer/storage?all=true, combine installation https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/, Claude Code https://code.claude.com/docs/en/setup.

## Part A — Yiyang's steps at Purdue

Run the blocks in order and look at each output. Running a block approves exactly that installation or copy. Keep existing files; back up a settings file before changing it. If an output differs from the expectation, continue with A8 and let the agent ask through a card.

### A0. Account (in progress)

1. Request for Privileges (R4P) form (highlighted sections) to physics@purdue.edu, cc cms-support@lists.purdue.edu and your local PI; the physics business office confirms by email.
2. ScienceIT (sciencehelp@purdue.edu, +1 765 494 4488, 8:00-17:00 EST) gives the Purdue career account and password (about one week).
3. The onboarding Google Form (linked on the Tier-2 accounts page) requests Tier-2 and Analysis Facility access; confirmation within 2 business days (contact cms-support otherwise).
4. Set up BoilerKey (Purdue two-factor). Login: `ssh -l <purdue_username> hammer.rcac.purdue.edu` (or `cms.rcac.purdue.edu`); SSH keys work if registered in advance. GPUs are on `gilbreth.rcac.purdue.edu`. The Analysis Facility (https://cms.geddes.rcac.purdue.edu/hub, "Purdue" login) is optional; Slurm jobs see only `/depot` and CVMFS, not the AF `/home` or `/work`, so the work lives on Hammer and `/depot`.

### A1. First login: check the environment

On the Hammer front-end:

```bash
whoami; hostname -f; head -3 /etc/os-release
myquota                                   # RCAC quotas: home, scratch, depot
echo "RCAC_SCRATCH=$RCAC_SCRATCH"
ls -ld /depot/cms/users/$USER /depot/cms/private/users/$USER 2>&1
slist                                     # Slurm accounts / queues you may use
ls -d /cvmfs/cms.cern.ch/el8_amd64_gcc13/cms/cmssw/CMSSW_16_1_0_pre4 /cvmfs/cms.cern.ch/el8_amd64_gcc13/cms/cmssw/CMSSW_16_0_0
ls /etc/grid-security/certificates | head -2; ls /etc/vomses 2>&1 | head -3
command -v tmux git curl apptainer singularity voms-proxy-init xrdcp gfal-copy sinteractive || true
```

Expected: Rocky Linux 8; one of the two depot directories exists and is yours (the Tier-2 tutorials use `/depot/cms/users/<user>/`, Jan uses `/depot/cms/private/users/<user>/`; if neither exists, ask cms-support for a depot directory); `slist` lists `cms` (and maybe `cms-express`, which Jan uses; it is not on the official queue list); both CMSSW releases exist for `el8_amd64_gcc13`. Write down every missing item for the agent.

For the commands of Part A only (not for `~/.bashrc`), set the paths of this shell:

```bash
DEPOT=/depot/cms/users/$USER              # or /depot/cms/private/users/$USER: the directory that exists
VVV=$DEPOT/CMS_Run3_VVV_Scouting          # analysis folder: repository, ntuples, results (as on Cannon)
SW=$DEPOT/software                        # global software: CMSSW areas (not inside the repository)
C=/n/holystore01/LABS/iaifi_lab/Lab/yiyangz/CMS_Run3_VVV_Scouting/ScoutingVVVTools   # repository on Cannon
```

### A2. Long-term global settings in `~/.bashrc`

Only global, long-term settings go into `~/.bashrc`; no project paths. These are the Cannon settings adapted to Purdue: the CMS environment, `~/.local/bin` (Claude Code), the grid-proxy alias, the `tw` tmux helper and `dirsize`. `X509_USER_CERT`/`X509_USER_KEY` are not exported: on Cannon, jobs that inherited them made the XRootD TLS client try the passphrase-protected key and fail; `voms-proxy-init` reads `~/.globus/usercert.pem` and `userkey.pem` by default.

~~~~bash
cp -p --no-clobber ~/.bashrc ~/.bashrc.before-vvv-$(date +%Y%m%dT%H%M%S)
cat >> ~/.bashrc <<'EOF'
# >>> global settings (from FASRC Cannon, 2026-10-13) >>>
source /cvmfs/cms.cern.ch/cmsset_default.sh
export PATH="$HOME/.local/bin:$PATH"
alias voms='voms-proxy-init --rfc --voms cms --valid 192:00'

# tw: attach the tmux session 'claude', or start it on this host (remembered in ~/host.txt).
unalias tw 2>/dev/null
tw() {
    if tmux has-session -t '=claude' 2>/dev/null; then
        tmux attach-session -t '=claude'
        return
    fi
    local here
    here=$(hostname -f)
    if [ -f ~/host.txt ] && [ "$here" != "$(cat ~/host.txt)" ]; then
        echo "The last 'claude' session was started on $(cat ~/host.txt): reattach it with" >&2
        echo "    ssh $USER@$(cat ~/host.txt)   then   tw" >&2
        read -r -p "Start a new 'claude' session here on $here instead? [y/N] " ans
        case "$ans" in y|Y) ;; *) return 1 ;; esac
        rm -f ~/host.bkp
        cp ~/host.txt ~/host.bkp || {
            echo "Failed to back up ~/host.txt to ~/host.bkp" >&2
            return 1
        }
    fi
    echo "$here" > ~/host.txt || {
        echo "Failed to write hostname to ~/host.txt" >&2
        return 1
    }
    echo "Creating tmux session 'claude' on $here"
    tmux new-session -s claude
}

# dirsize [dir]: total size of the files below dir (EOS metadata for CERN EOS paths).
dirsize() (
    set -o pipefail
    export LC_ALL=C
    local p remote host='' bytes='' kind initial rest
    p=$(cd -- "${1:-.}" && pwd -P) || exit 1
    remote="$p"
    if [[ "$p" =~ ^/eos/(home|project)-([a-z])(/.*)?$ ]]; then
        kind="${BASH_REMATCH[1]}"; initial="${BASH_REMATCH[2]}"; rest="${BASH_REMATCH[3]}"
        host="eos${kind}-${initial}.cern.ch"
        [[ "$kind" == home ]] && kind=user
        remote="/eos/$kind/$initial$rest"
    elif [[ "$p" =~ ^/eos/(user|project)/([a-z])(/.*)?$ ]]; then
        kind="${BASH_REMATCH[1]}"; initial="${BASH_REMATCH[2]}"
        [[ "$kind" == user ]] && kind=home
        host="eos${kind}-${initial}.cern.ch"
    elif [[ "$p" =~ ^/eos/(cms|atlas|lhcb|alice)(/|$) ]]; then
        host="eos${BASH_REMATCH[1]}.cern.ch"
    fi
    if [[ -n "$host" ]] && command -v eos >/dev/null 2>&1; then
        bytes=$(timeout 15s eos "root://$host" fileinfo "$remote" 2>/dev/null |
            sed -nE 's/.*Treesize[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p') || bytes=''
    fi
    if [[ ! "$bytes" =~ ^[0-9]+$ ]]; then
        bytes=$(find "$p" -type f -printf '%s\n' | awk '{s += $1} END {printf "%.0f\n", s}') || exit 1
    fi
    numfmt --to=iec-i --suffix=B --format='%.2f' "$bytes"
)
# <<< global settings <<<
EOF
source ~/.bashrc
~~~~

If A1 showed no `/etc/grid-security/certificates` or `/etc/vomses`, tell the agent (it will propose `--certdir /cvmfs/grid.cern.ch/etc/grid-security/certificates --vomses /cvmfs/grid.cern.ch/etc/grid-security/vomses` for the alias).

### A3. Analysis folder and repository

```bash
mkdir -p "$VVV/ntuples" "$VVV/results" "$SW"
git clone --branch fasrc-campaign-20261008 https://github.com/yiyangzha/ScoutingVVVTools.git "$VVV/ScoutingVVVTools"
cd "$VVV/ScoutingVVVTools" && git log -1 --oneline && git status --short | head
```

### A4. Global software (outside the repository)

Analysis environment (ROOT, CMSSW Python with xgboost/uproot/scikit-learn/scipy/zstandard, correctionlib), as Jan uses it:

```bash
cd "$SW"
export SCRAM_ARCH=el8_amd64_gcc13
cmsrel CMSSW_16_1_0_pre4
cd CMSSW_16_1_0_pre4/src && cmsenv
root-config --version; gcc --version | head -1
python3 -c "import xgboost, uproot, sklearn, scipy, zstandard, pandas, matplotlib, mplhep, correctionlib; print('xgboost', xgboost.__version__, 'uproot', uproot.__version__, 'correctionlib', correctionlib.__version__)"
python3 -c "import correctionlib, os; print(os.path.dirname(correctionlib.__file__))"   # must lie in CVMFS, not ~/.local
```

combine (official recipe, CombinedLimit v11.1.0 in CMSSW_16_0_0; build it in an interactive job if the front-end limits CPU, e.g. `sinteractive -A <account> -n 8 -t 02:00:00`):

```bash
cd "$SW"
export SCRAM_ARCH=el8_amd64_gcc13
cmsrel CMSSW_16_0_0
cd CMSSW_16_0_0/src && cmsenv
git -c advice.detachedHead=false clone --depth 1 --branch v11.1.0 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
scramv1 b clean; scramv1 b -j8
combine --help | head -3
```

### A5. Claude Code and its settings

```bash
curl -fsSL https://claude.ai/install.sh | bash          # launcher in ~/.local/bin, binary in ~/.local/share/claude
claude --version
mkdir -p ~/.claude && chmod 700 ~/.claude
for f in settings.json CLAUDE.md; do [ -e ~/.claude/$f ] && cp -p --no-clobber ~/.claude/$f ~/.claude/$f.before-vvv-$(date +%Y%m%dT%H%M%S); done
scp yiyangz@login.rc.fas.harvard.edu:.claude/settings.json yiyangz@login.rc.fas.harvard.edu:.claude/CLAUDE.md ~/.claude/
chmod 600 ~/.claude/settings.json ~/.claude/CLAUDE.md
claude          # log in once (browser link), then /exit
```

(If the FASRC login from Purdue fails, push from Cannon instead: `scp ~/.claude/settings.json ~/.claude/CLAUDE.md <purdue_username>@hammer.rcac.purdue.edu:.claude/`.) Do not copy `~/.claude/.credentials.json`, histories, sessions or plugins.

### A6. Grid certificate and proxy

```bash
umask 077
mkdir -p ~/.globus && chmod 700 ~/.globus
scp yiyangz@login.rc.fas.harvard.edu:.globus/usercert.pem yiyangz@login.rc.fas.harvard.edu:.globus/userkey.pem ~/.globus/
chmod 400 ~/.globus/usercert.pem ~/.globus/userkey.pem
openssl x509 -in ~/.globus/usercert.pem -noout -subject -enddate
voms                                                     # proxy /tmp/x509up_u$(id -u) on this front-end
voms-proxy-info --vo --timeleft --actimeleft
```

`run.py --site purdue --slurm` copies `/tmp/x509up_u<uid>` to `/depot/cms/users/$USER/` for the jobs; the agent checks and adapts that path (Part B5).

### A7. Files from Cannon

The pileup reference histograms (authorized external inputs, git-ignored `*.root`):

```bash
cd "$VVV/ScoutingVVVTools"
mkdir -p selections/weight/pileup tmp/lxplus_full_20261001/inputs/pileup
scp "yiyangz@login.rc.fas.harvard.edu:$C/selections/weight/pileup/dataPileupHistogram-2024CDEFGHI_Golden-*.root" selections/weight/pileup/
cp -p --no-clobber selections/weight/pileup/dataPileupHistogram-2024CDEFGHI_Golden-*.root tmp/lxplus_full_20261001/inputs/pileup/
sha256sum selections/weight/pileup/dataPileupHistogram-2024CDEFGHI_Golden-*.root
```

Expected SHA256: `66000ub` 02be96ef021370cb8c6e6d66e49bc40bf3e9e9c75711b7d7491c70d7c2f97454, `69200ub` 5e379a2004788f693187f6a579a81f2530a27a00482e39b559337d026c40fb93, `72400ub` 3b85a78a4a1ee929edccf9b785e5c50095022cdd74e5a29539fba4842dfcb253 (the official copies are linked in README.md).

Optional, for comparison only (Cannon PU products of 82 samples; several Purdue-hosted samples skipped unreadable files there), and Jan's audit page:

```bash
mkdir -p "$VVV/results/cannon_20261008_pileup"
scp "yiyangz@login.rc.fas.harvard.edu:$C/selections/weight/pileup/pileup_*" "$VVV/results/cannon_20261008_pileup/"
scp "yiyangz@login.rc.fas.harvard.edu:$C/tmp/VVV_Sensitivity_Audit.pdf" tmp/
```

### A8. Start the agent

```bash
tw                                        # tmux session 'claude' (detach: Ctrl-b d)
cd "$VVV/ScoutingVVVTools"
claude
```

Startup instruction for the new agent:

> Read tmp/prompt_purdue_20261013.md completely and execute Part B. Read AGENTS.md, README.md and the records in tmp/fasrc_20261008/ completely, and Jan's selections/BDT/study_boosted3/significance_review.md. Continue the campaign at Purdue (Hammer, my Purdue account) from the registered MC and the 2024G_UParT_v6 data. The analysis folder is the parent of this repository; global software is in <DEPOT>/software (CMSSW_16_1_0_pre4, CMSSW_16_0_0 with combine v11.1.0). Every matter needing my input is a visible Chinese Plan-style question card, including confirmations and download/installation/deletion permission; verify the card at the first decision. Adapt the server integration completely; ask before any physics or analysis-logic change. Keep working, monitoring jobs, until the task is complete, a card awaits my answer, or an unsolvable problem blocks progress.

(Replace `<DEPOT>` with the directory of A1.)

### A9. Later logins

`ssh` to the front-end shown by `tw` (or any, `tw` tells you), `tw`, renew the proxy with `voms` when `voms-proxy-info --timeleft` is short, and in tmux `cd <repo>; claude --continue`, then tell the agent "继续".

## Part B — instructions for the new agent

### B1. Objective

Complete the analysis campaign at Purdue: PU weights → ttbar control region → JMS/JMR fit → analysis conversion (five trees `boosted3`, `boosted2_resolved`, `boosted2_unresolved`, `boosted2_leptonic_1lep`, `boosted2_leptonic_2lep` plus the eight MC jet variations) → mix → k-fold BDT training (5 rotations per tree) and k-fold scoring → inclusive systematics → SR optimization → per-SR ABCD QCD → per-SR systematics → Data/MC and class-shape plots → combine expected significance and limits (combined, class, sample, per channel; MC-true and ABCD QCD), plus the studies of B7. Inputs: the registered MC of `src/sample.json` in the BDT `class_groups` and the ttbar-CR list, and only the data `2024G_UParT_v6`; expected sensitivity for 255.8 fb^-1 with the 2024 C-I PU profile. Every stage is redone at Purdue; no Cannon product is reused (requirement 40), apart from the optional comparison copies of A7. Completion means validated real outputs, not submissions or exit codes.

### B2. Rules

1. Requests for input: visible Chinese `AskUserQuestion` cards with evidence, concrete options and a recommendation with its reason; one round at a time; restate and record the answer. Ordinary messages report progress only.
2. Keep executing until the objective is reached; stop only while a card awaits an answer or for an unsolvable blocker. Monitor jobs every few minutes with bounded waits; keep timestamped records.
3. Preserve every file: no deletion (intermediates, caches, failed outputs, program cleanup) without an explicit card; collect deletion proposals for one final decision. Approved program cleanups: the three convert_branch cleanups (thread temp files after their verified fast merge, the hidden partial output of a failed merge, a batch's old `.meta` before reprocessing); combine `keep_work` stays true.
4. Write scope: the repository; the analysis folder (`ntuples/`, `results/`); the global software folder only for the A4 areas and approved installations; `$RCAC_SCRATCH` (intermediates, 60-day purge) and `/tmp/$USER`; Purdue EOS `/store/user/<user>` only after a card; `~/.claude` is managed by Claude Code. Everything else is read-only (Jan's `/depot/cms/private/users/schul105` included).
5. Git is read-only (status, diff, log, show, check-ignore); commits, pushes and `.gitignore` changes need an explicit card. Report the tracked/ignored state of new files at the end.
6. No installation or download without a card naming source, version, destination, command and purpose. Remote DAS/XRootD reads are approved for the MC in scope and `2024G_UParT_v6` only.
7. Execution: the Hammer front-end is for light work (editing, submission, monitoring); compilation of the campaign tools, environment checks and all processing run in Slurm jobs. Normal campaign compilation, Slurm submission, execution, monitoring and recovery are authorized. Pilot every stage on a small subset and check the result before the full run.
8. Credentials: never print or copy key/proxy contents; check `--timeleft`/`--actimeleft` before long runs; ask for renewal by card.
9. At most two subagents at a time, current model, effort by difficulty. Work slowly and carefully; fix root causes; report unrelated problems instead of fixing them; physics or analysis-logic changes need a card first.
10. Records: create `tmp/purdue_<YYYYMMDD>/{REQUIREMENTS.md,PLAN.md,LOG.md,progress.json}` (English, untracked), carry over requirements 1-66 of `tmp/fasrc_20261008/REQUIREMENTS.md` by exact reference (Cannon-specific paths, Slurm accounts, scratch and proxy rules there are replaced by this prompt; the physics decisions stay), record commands, job IDs, versions, hashes, provenance and answers with real timestamps. Keep `tmp/fasrc_20261008/` unchanged. At every small-stage checkpoint reread this prompt, AGENTS.md, README.md and the complete records. Keep AGENTS.md (section 6) and README.md in sync with code changes.

### B3. Confirmed decisions (full text in `tmp/fasrc_20261008/REQUIREMENTS.md`)

- MC weights (24, 54): w = L·σ·genWeight·weight_pu / S × N_tree/N_read, S = `sum_genweight_pu` from the convert merge (`selections/mc_weight_common/mc_weights.py`), everywhere (Jan's `gen_weight.py` is unused); training weights |w|·Σw/Σ|w| per sample and split; a split with a non-positive signed sum warns and gets zero training weight. After the conversion, compare S with `raw_entries × genweight_mean × weight_pu_mean` (Jan's former normalisation) per sample and report.
- Theory (55): acceptance-only theory variations (`generator_sums.py`, `generator_sums_json`) plus per-process `xsec_<process>` lnN from `combine/xsec_unc.json` (values "PROPOSED, to be reviewed"); VV normalised to NNLO QCD + NLO EW.
- Samples (56): Jan's `class_groups`; the QCD samples with HT < 400 GeV (`qcd_ht40to70`, `qcd_ht70to100`, `qcd_ht100to200`, `qcd_ht200to400`; pileup double counting) are in the training and excluded from every physics yield (`exclude_samples` of the SR scan, qcd_est and plotting configs).
- Production settings (61-63): k-fold 5 with Jan's `kfold_finedecor` settings; SR scan with `zero_count_floor`, `exclude_samples`, `abcd_uncertainty` (msoftdrop_1 control window 20-50 GeV, non-closure 0.20), `min_bkg_entries` 0, branch-and-bound without node cap and 14400 s; qcd_est `abcd_mode per_sr` with the complement sideband, the same window, non-closure and exclusion.
- SR optimizer (57): the single optimizer `signal_region_hist.py` (mode 3) described in its docstring and in AGENTS.md: Z_A per region with V = correlated systematic rows (lumi, trigger, theory_pdf, theory_scale per class, PS ISR/FSR, pileup, JES, JER, JMS, JMR, xsec per class; inclusive ratios) + MC statistics + zero-count floor + ABCD variance; in ABCD mode the QCD of a box is the prediction C_box × TF; exact-N branch-and-bound on Σ Z_A², then a joint stage maximising Z_joint (profile-likelihood Asimov of all regions with the correlated rows and the ABCD non-closure), fine refinement kept only if Z_joint does not drop, boundaries in empty score gaps (scan and control events preserved).
- Earlier decisions stay: AK4 cleaning against the two leading AK8 jets (21); mix then train on `{sample_group}_mixed` (22); ttbar-CR VV class = the eight exclusive VV samples (27); reference checks strict, report measured differences before changing tolerances (29); Data/MC display projected from 27.51 to 255.8 fb^-1, approved but not implemented (30); systematic ratios on the full converted samples (31); weight.C skips unreadable MC files and lists them (47); keep the registered MC versions and skip unreadable files (45).

### B4. State of the code (nothing of the 2026-10-13 changes has been run or compiled)

- Merge of Jan's branch: his k-fold training/scoring (`kfold.py`, `score_kfold.py`, `run.py` mode 2 `--slurm` one job per tree and rotation), train/eval/test budgets, per-SR ABCD (`qcd_est.py`, Jan's `combine.C`), `generator_sums.py`, `xsec_unc.json`, two-pass blinded `data_mc.py`, `sr_sensitivity.py`, study scripts; with our weights, strict reference checks (also in `score_kfold.py`), `mktree` metadata, thread affinity, site profiles, signed-sum convert merge.
- New optimizer code in `signal_region_hist.py` (B3). Its functions `background_syst_shifts`, `Ctx`, `top_intervals_on_axis`, `_AxisIndex`, `dedupe_by_event_mask`, `place_boundaries`, `region_components`, `joint_z`, `joint_refine_selection` are unvalidated.
- `--site purdue` (default) is Jan's behaviour: tools on the PATH of the submitting shell (`cmsenv` of `CMSSW_16_1_0_pre4`), Slurm account `cms-express`, proxy copied from `/tmp/x509up_u<uid>` to `/depot/cms/users/$USER/`, local binaries removed at exit (hashed binaries kept with `--slurm`).
- Not yet created: the Purdue campaign configs (input/output paths). The inclusive systematics configs exist as `selections/<syst>/config_inclusive.json` (no `signal_region_csv`, output `output/inclusive/`, read by the SR objective's `background_syst_jsons`); their campaign copies need the Purdue `input_root`.
- An independent static review on Cannon found no remaining crash in the merged code after its fixes; it could not run anything.

### B5. Purdue integration (state the plan first; keep the fasrc and ihep profiles working)

1. Verify the environment of A1/A4 in a Slurm job (CMSSW_16_1_0_pre4 el8: GCC, ROOT, correctionlib, the Python packages; CMSSW_16_0_0 combine v11.1.0) and record the versions. XGBoost of this CMSSW is 1.7.x (Cannon's pixi had 2.1.4): check that `train.py` and `model_io.py` run with it (Jan trained with it).
2. Proxy for jobs: check that `/depot/cms/users/$USER/` exists and is writable; otherwise adapt the `purdue` profile (for example an explicit proxy under the depot directory, like `fasrc`'s `explicit` mode) and record it; Slurm jobs must see the proxy (`/depot`). Jobs must not carry `X509_USER_CERT`/`X509_USER_KEY`.
3. Slurm: confirm the account and partition with `slist` (`cms` official; `cms-express`/`hammer-nodes` as in Jan's scripts) and the resource limits; `run.py --slurm-account` and `--slurm-partition` select them.
4. combine: run mode 7 with `combine`/`combineCards.py` of CMSSW_16_0_0 on the PATH (a site wrapper like `fasrc`'s if needed); check `combine.C` against v11.1.0 (Jan used `combine` from his own area).
5. Paths: campaign configs (untracked, next to the checked-in ones; the converter needs the sibling `branch.json`/`selection.json`) with absolute paths into the analysis folder; outputs of each step in its default directory (overwriting the tracked `jms_jmr_results.json` and `pileup_syst_yields.json` is approved).
6. Storage (card): measure the per-file sizes in the pilots, extrapolate (nominal + 8 variations, mixed copies, CR, batch temp files) and propose where intermediates (`$RCAC_SCRATCH`) and durable ntuples (`$VVV/ntuples` on depot, up to 1 TB; Purdue EOS `/store/user/<user>`, 10 TB, via `root://eos.cms.rcac.purdue.edu`) go.
7. Data access (card, requirement 59): `2024G_UParT_v6` (6.26 TB, 66,228 files) and 32 MC samples owned by yiyangz (1.76 TB) are only at IHEP (no Rucio record). Measure from Purdue the read rates through `root://xcache.cms.rcac.purdue.edu//`, the global redirector and parallel `xrdcp --streams` copies, estimate the conversion time and the storage of a staged copy, and ask. The 52 jschulte MC samples are at Purdue (`root://eos.cms.rcac.purdue.edu//`, `/eos/purdue`); re-test the Cannon-unreadable files (`tmp/prompt_purdue_af_20261008.md` B5 table, `selections/weight/pileup/pileup_*_skipped.txt` on Cannon).

### B6. Execution plan and acceptance

1. Inventory, records, first card; environment checks in a job (B5.1); syntax and import checks of the changed Python files in a job.
2. Integration (B5) with a stated plan; pilot of each C++ tool.
3. Data-access measurement and decision (B5.7).
4. PU (all MC in scope; pilot first), ttbar CR (MC + data; pilot first), JMS/JMR fit at the real CR statistics.
5. Analysis conversion (nominal + 8 variations), pilots of a signal, QCD, ttbar and data batch (time, size, entries, `sum_genweight*` vs Runs `genEventSumw`), full production, validation of every batch/merge/sidecar; `generator_sums.py` for the theory samples.
6. Mix; k-fold training (5 rotations × 5 trees; `run.py 2 --slurm`); `submit_score_kfold.sh` scoring with `--branches` listing every branch the downstream steps read from the cache beyond the thresholds and decorrelation branches (at least the union of the `a_region_shape_branches` of `background_estimation/config_<tree>.json`, else qcd_est stops with a KeyError), with the Slurm account of B5.3; reference checks.
7. Inclusive systematics (new inclusive configs), SR optimization per tree (runtime measured on boosted3 first, requirement 63), qcd_est per_sr, per-SR systematics (`sr_csv_hash` provenance), Data/MC (implement the approved projection) and class shapes.
8. combine: all scenarios, per channel, MC-true and ABCD; inspect cards, fits and quantiles.
9. Studies of B7; final verification, deletion proposals (B8), git status/check-ignore of new files.

### B7. Studies and open items

1. Validation of the new optimizer: (a) `joint_z` for one region without correlated rows against Z_A (close, Gaussian vs Poisson auxiliary constraint) and against a one-SR combine card; (b) Z_joint of the selected set against the full combine result of the same SRs; (c) runtime of deduplication, branch-and-bound and the joint stage on boosted3 with the k-fold inputs; (d) the residual optimisation bias of the k-fold SRs (Jan's open caveat): choose the SRs on the events with even event number and evaluate them with combine on the odd ones, and vice versa; report the bias by card.
2. ABCDEF (requirement 51): measure the non-closure of the per-SR ABCD in data instead of taking 0.20 from MC. Proposal to evaluate: a second msoftdrop_1 band E/F in the fail region (Jan's `abcd_viability.py` found the band above the window 1.5-3× harder to close, so a band below the 20-50 GeV control window, e.g. 0-20 GeV, or a split of the window, is the candidate; check the decorrelation training window `decor_msoftdrop_training_max` and msoftdrop ≤ 0 events). Predict C from E: C_i^pred = (E_i - E_i^nonQCD)·(D - D^nonQCD)/(F - F^nonQCD) in data (non-QCD from MC), k_i = (C_i^data - C_i^nonQCD)/C_i^pred, and use f_nc = max(|1 - k|, σ_k) per SR or for the SR union (whichever the data statistics support) as `abcd_per_sr_nonclosure`. All regions lie below the signal mass window, so the window stays blinded (confirm by card). Steps: validate E→C closure in MC first (`abcd_closure.py`, per msd bin, by fold and by sample); then data. Code: qcd_est.py (E/F masks, a data loader with k-fold scores by event number as in `data_mc.py`, `validation/*` bundles, the derived non-closure), optionally combine.C (E/F bins in the control card). With the data-driven non-closure, QCD MC statistics no longer enter the non-closure; the SR optimizer's QCD term then uses that value. Plan by card before coding.
3. MC statistics of the non-QCD backgrounds (requirement 51): the optimizer already includes their Σw² and floors; after the SRs are fixed, make a variance budget: per SR and sample the share of Σw² in the total background variance, the VV share in particular (NLO FxFx samples, 19-22 % negative weights, N_eff = N(1-2f)²), and the number of additional events of each sample that would halve its contribution; propose a targeted VV (or other) sample request with its expected gain by card. No request is made without approval.
4. `xsec_unc.json` values (VVV 10 % placeholder, V+jets 10 %) and the VV uncertainty: review with Yiyang.
5. `boosted2_leptonic_2lep` with k-fold: samples with fewer entries than blocks (Jan dropped the HT < 400 GeV QCD, `WZto2Nu2Q` and `WZto4Q` for this tree) stop train.py on an empty split; check and propose (card).
6. `data_mc.py` score validation (requirement 29): the two-pass design does not call `_compare_score_reference`; propose a check against the k-fold score cache (card).
7. Luminosity: confirm 255.8 fb^-1 for the analysed 2024 G/H/I data.
8. Minor items of the Cannon review: `train.py` `_apply_class_budget` shares a class budget by `xsec × N_tree / raw_entries` (an allocation heuristic only; the weights are exact), which is less uniform for NLO samples than `xsec × N_tree / S × mean raw_w`; `np.histogram` in `top_intervals_on_axis` closes the last bin at 1.0 while `rect_mask` treats an upper edge at 1.0 as open (relevant only for merged score axes summing slightly above 1); the old study configs `config_hist_decorr3_tuned.json`, `config_minbkg50_fat3_v2.json`, `config_tailaware_fat2.json` and `selections/BDT/config_tailaware.json` lack the now required keys and stop with errors. Report, fix only by card.

### B8. Cannon leftovers (not deleted; for the final deletion card only)

On Cannon (`/n/holystore01/LABS/iaifi_lab/Lab/yiyangz/CMS_Run3_VVV_Scouting/ScoutingVVVTools` and `/n/netscratch/iaifi_lab/Lab/yiyangz/CMS_Run3_VVV_Scouting/`): pilot outputs `fasrc_20261008/pilot/`, the pixi cache and tmp on scratch, the repository `.pixi/` environments, `CMSSW_16_1_0_pre4/`, built binaries under `build/fasrc/bin/` and `selections/weight/weight_*`, 82 PU CSV/PDF/skipped lists in `selections/weight/pileup/`, PU job logs in `selections/weight/`, diagnostics under `build/fasrc/fasrc_20261008/`, the untracked `config_fasrc*.json`/`config_ihep*.json`, and the private proxy backup `build/fasrc/credentials/cms-20261008T101031-0400/`. On IHEP (`/ihepfs/cms_thu/yiyangzhao/Research/claude/`): the repository clone, its CMSSW area, the ttbar-CR batch outputs under `ntuples/fasrc_20261008/` (merges never run) and `build/`. Nothing runs on Cannon or IHEP.
