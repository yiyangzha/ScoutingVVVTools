# CMS Run 3 VVV scouting: complete fresh rerun on FASRC Cannon

Prepared on 2026-10-05 from the current ScoutingVVVTools repository. Part A is for Yiyang to execute manually; Part B is the new agent's task. The handoff is in English. Every request for Yiyang's input must use a visible Plan-style question card in Chinese, including confirmations, installation/download permission, deletion permission, and problems. Ordinary messages are for information that requires no reply.

The source checkout inspected for this update is:

- Repository: git@github.com:yiyangzha/ScoutingVVVTools.git
- Branch: main
- Commit: fbe3bc40214ede6366d999605d5b06b361679179 (PR #10 merged).
- Source path: /eos/home-y/yiyangz/CMS_Run3_VVV_Scouting/ScoutingVVVTools
- Historical handoff preserved as tmp/prompt_lxplus_20261001_backup.md.

This document supersedes the lxplus execution instructions. Historical records preserve earlier decisions and evidence; their old paths, environment observations, repair lists, and authorizations must be reconciled against the current source. This preparation did not run, compile, install, or submit the analysis on Cannon.

## Part A — Yiyang's setup commands

Execute the steps in order. These commands are instructions for you; the agent preparing this document has not executed them. Check each result before continuing. Executing an installation/copy block approves that specific action, rather than every future installation. Keep existing files and settings; use a new backup name before replacing an existing settings file. If something differs from these expectations, have the agent ask through a Chinese Plan-style card.

### A0. Log in and check the agreed storage

On your own computer, connect through the FASRC VPN when required:

```bash
ssh yiyangz@login.rc.fas.harvard.edu
```

On the Cannon login node:

```bash
hostname -f
export IAIFI=/n/holystore01/LABS/iaifi_lab/Lab/yiyangz
ls -ld "$IAIFI"
id
df -h "$IAIFI"
```

The agreed path contains `Lab`, with precisely that capitalization. Verify it exists and is writable. A missing lab allocation is a problem to resolve, rather than a reason to choose another storage root. The user has approved this layout:

```text
/n/holystore01/LABS/iaifi_lab/Lab/yiyangz/CMS_Run3_VVV_Scouting/
  ScoutingVVVTools/          code, campaign configs/records and project environments
  ntuples/<campaign>/       reusable fresh analysis and ttbar-CR ntuples
  results/<campaign>/       reusable calibrations, models, SRs, plots, systematics, combine results
/n/netscratch/iaifi_lab/Lab/yiyangz/CMS_Run3_VVV_Scouting/<campaign>/
                            intermediate files and I/O-intensive work
```

Ntuples must be preserved in Lab storage because they may be reused. Scratch is authorized for intermediate processing, and deletion still requires a question card and Yiyang's explicit confirmation. Keep input manifests, metadata sidecars and important recovery state in Lab as well. Lab storage has no backup; scratch has an age-based purge. Check current limits and available capacity before production.

Codex follows the account-wide setup pattern in `Research/codex/slopbench_harness/work/lep_calib/FASRC_HANDOFF.md` Part A (tools in A2, global settings in A6): its executable belongs in the account's shell PATH, and its shared settings belong in `$HOME/.codex/`. These are shared by this FASRC account across repositories and nodes that share home storage.

### A1. Reuse the already installed tools; clone this repository

The lep_calib handoff already arranged pixi, GitHub access, tmux and the shell PATH. Reuse them; do not reinstall pixi, Claude, gh, or recreate SSH keys.

Persist the stable account/project paths in `~/.bashrc`, preserving the existing lep_calib setup. The block below is added once; on subsequent runs inspect and reuse it. The initial A0 export only bootstraps the storage check. Transfer sources, setup timestamps and the selected expiring CMS proxy remain specific to the transfer/campaign session.

```bash
if grep -q -F '# >>> scouting-vvv (FASRC) >>>' "$HOME/.bashrc"; then
  sed -n '/^# >>> scouting-vvv (FASRC) >>>$/,/^# <<< scouting-vvv (FASRC) <<<$/p' "$HOME/.bashrc"
else
  cp -p --no-clobber "$HOME/.bashrc" \
    "$HOME/.bashrc.before-vvv-$(date +%Y%m%dT%H%M%S)"
  cat >> "$HOME/.bashrc" <<'EOF'
# >>> scouting-vvv (FASRC) >>>
export IAIFI=/n/holystore01/LABS/iaifi_lab/Lab/yiyangz
export VVV_PROJECT="$IAIFI/CMS_Run3_VVV_Scouting"
export VVV_REPO="$VVV_PROJECT/ScoutingVVVTools"
export PIXI_CACHE_DIR="$IAIFI/.cache/pixi"
export PATH="$HOME/.local/bin:$HOME/.pixi/bin:$PATH"
# <<< scouting-vvv (FASRC) <<<
EOF
fi
source "$HOME/.bashrc"
command -v pixi claude git tmux rsync ssh curl
readlink -f "$(command -v claude)"
pixi --version
type tw
ssh -T git@github.com
```

GitHub normally reports successful SSH authentication while returning exit status 1. If your existing key needs unlocking, use its existing ssh-agent/key setup. Do not copy the lxplus SSH private key.

The reference handoff's A2 names `~/.local/bin/claude` as Claude's entry point; A1 places the pixi cache under `$IAIFI/.cache/pixi`. Its home directory is shared across login nodes. Inspect the resolved Claude path above to establish the actual storage location: a symlink may point elsewhere. The documented home-based layout is the baseline for Codex in A3. If the actual Claude installation resolves under `/n/holystore01/LABS/iaifi_lab/Lab/yiyangz`, use that same account-wide storage root for Codex, and settle its exact installation command and permanent PATH before executing A3.

If `$VVV_REPO` already exists, inspect it and preserve its edits; do not run the clone block over it. Otherwise:

```bash
mkdir -p "$VVV_PROJECT/ntuples" "$VVV_PROJECT/results"
cd "$VVV_PROJECT"
git clone --branch main git@github.com:yiyangzha/ScoutingVVVTools.git ScoutingVVVTools
cd "$VVV_REPO"
git status --short
git log -1 --oneline
git rev-parse HEAD
mkdir -p build/fasrc/reference tmp
```

A fresh clone should have a clean working tree before private records are copied. Record the exact commit. If GitHub has advanced beyond the source commit above, use the checked-out current source and have the new agent reconcile the differences before production; preserve the physics decisions.

The one-time clone is explicitly requested setup. Once the agent starts, git is read-only: no fetch, checkout, reset, stash, commit or push.

### A2. Copy the prompt, historical evidence, and IAIFI guide from lxplus

Use the current repository, not `ScoutingVVVTools_un`. No old converted ntuples or analysis results are to be imported as fresh campaign products.

```bash
export VVV_LXPLUS=yiyangz@lxplus.cern.ch
export VVV_SOURCE=/eos/home-y/yiyangz/CMS_Run3_VVV_Scouting/ScoutingVVVTools
export VVV_HARNESS=/eos/home-y/yiyangz/codex/slopbench_harness
cd "$VVV_REPO"

rsync -a --ignore-existing --info=progress2 \
  "$VVV_LXPLUS:$VVV_SOURCE/tmp/prompt.md" \
  "$VVV_LXPLUS:$VVV_SOURCE/tmp/prompt_lxplus_20261001_backup.md" \
  "$VVV_LXPLUS:$VVV_SOURCE/tmp/lxplus_workflow_20261001_plan.md" \
  "$VVV_LXPLUS:$VVV_SOURCE/tmp/lxplus_workflow_20261001_log.md" \
  "$VVV_LXPLUS:$VVV_SOURCE/tmp/lxplus_workflow_20261001_progress.json" tmp/

mkdir -p tmp/lxplus_full_20261001/inputs/pileup tmp/lxplus_full_20261001/metadata
rsync -a --ignore-existing --info=progress2 \
  "$VVV_LXPLUS:$VVV_SOURCE/tmp/lxplus_full_20261001/inputs/pileup/" \
  tmp/lxplus_full_20261001/inputs/pileup/
rsync -a --ignore-existing --info=progress2 \
  "$VVV_LXPLUS:$VVV_SOURCE/tmp/lxplus_full_20261001/metadata/" \
  tmp/lxplus_full_20261001/metadata/

rsync -a --ignore-existing \
  "$VVV_LXPLUS:$VVV_HARNESS/work/IAIFI_Computing_Cannon-Resources-for-Investigators.pdf" \
  "$VVV_REPO/build/fasrc/reference/"
rsync -a --ignore-existing \
  "$VVV_LXPLUS:$VVV_HARNESS/work/lep_calib/FASRC_HANDOFF.md" \
  "$VVV_REPO/build/fasrc/reference/lep_calib_FASRC_HANDOFF.md"

sha256sum tmp/lxplus_full_20261001/inputs/pileup/*.root
git status --short
git check-ignore -v tmp/prompt.md build/fasrc/reference/lep_calib_FASRC_HANDOFF.md
```

For rsync, a source directory ending in `/` copies its contents. These commands preserve existing destination files; they use no deletion option. On a repeat transfer, explicitly reconcile any differing existing file, especially the prompt: `--ignore-existing` deliberately leaves it intact. [FASRC rsync](https://docs.rc.fas.harvard.edu/kb/rsync/)

The current source's `.gitignore` ignores `build/` but does **not** ignore `tmp/`. Expect private records to appear as untracked. Do not commit them. Ask through a card before changing `.gitignore`; the historical claim that `/tmp/` was added belongs to the other checkout.

### A3. Install Codex for the FASRC account; copy global settings and question-card feature

Run this one-time setup on the Cannon login node, independently of the repository, after checking Claude's resolved installation path in A1. The commands below implement the reference handoff's shared-home layout. This is a server account-wide installation: Codex is available from every project through the permanent shell PATH, and `$HOME/.codex/config.toml` supplies the account's defaults. Install Codex only if it is absent. This uses the current official standalone installer and avoids installing Node/npm just for Codex. Load the permanent settings established in A1.

```bash
cd "$HOME"
source "$HOME/.bashrc"
```

Check any existing `CODEX_HOME` override before continuing: this setup uses the standard `$HOME/.codex` location. Reconcile an existing override with the account's shell configuration first, so Codex reads the global files installed below.

```bash
if command -v codex >/dev/null 2>&1; then
  codex --version
else
  curl -fsSL https://chatgpt.com/codex/install.sh | sh
fi
source "$HOME/.bashrc"
command -v codex
readlink -f "$(command -v codex)"
codex --version
```

Installer reference: [official Codex CLI documentation](https://learn.chatgpt.com/docs/codex/cli). The inspected lxplus standalone release was 0.160.1. Record the actual Cannon version and check feature support rather than assuming a version match.

Transfer configuration and global instructions to a private, uniquely named setup directory under the account's Codex home, excluding authentication/session caches. The transfer and backups belong to the account-wide setup, independently of any analysis checkout:

```bash
umask 077
mkdir -p "$HOME/.codex/setup"
chmod 700 "$HOME/.codex" "$HOME/.codex/setup"
export CODEX_SETUP_TAG=$(date +%Y%m%dT%H%M%S)
export CODEX_TRANSFER=$(mktemp -d "$HOME/.codex/setup/lxplus-$CODEX_SETUP_TAG-XXXXXX")
rsync -a --ignore-existing \
  "$VVV_LXPLUS:/afs/cern.ch/user/y/yiyangz/.codex/config.toml" \
  "$CODEX_TRANSFER/config.lxplus.toml"
rsync -a --ignore-existing \
  "$VVV_LXPLUS:/afs/cern.ch/user/y/yiyangz/.codex/AGENTS.md" \
  "$CODEX_TRANSFER/AGENTS.lxplus.md"
rsync -a --ignore-existing \
  "$VVV_LXPLUS:/afs/cern.ch/user/y/yiyangz/.codex/rules/" \
  "$CODEX_TRANSFER/rules_lxplus/"
```

The archived rules include permissions tied to unrelated lxplus projects, including old deletion permissions. Keep them as a reference; do not activate them as this campaign's permissions. Do not copy `auth.json`, API keys, session histories, or plugins that would download themselves.

Produce the account-wide Cannon config from the copied settings. Omit the source's lxplus-specific project trust sections while retaining the model, reasoning effort, retry/timeout settings, telemetry settings, and the question-card feature. Each Cannon repository's trust is established separately when first opened. Global configuration applies across projects; a project-local `.codex/config.toml` supplies project overrides. [Official Codex configuration scope](https://learn.chatgpt.com/docs/config-file/config-basic)

```bash
awk '
  /^\[projects\./ {skip=1; next}
  /^\[/ {skip=0}
  !skip {print}
' "$CODEX_TRANSFER/config.lxplus.toml" > "$CODEX_TRANSFER/config.fasrc.toml"
grep -n -A 3 '^\[features\]' "$CODEX_TRANSFER/config.fasrc.toml"
```

Expect the following single feature table to retain this setting:

```toml
[features]
default_mode_request_user_input = true
```

The feature exists in the [official Codex configuration schema](https://developers.openai.com/codex/config-schema.json). Avoid adding a duplicate `[features]` table. If the copied file lacks the setting, add it to the existing table or create one table if absent. Restart Codex after configuration changes. A settings line is only setup evidence; the new session must actually display a working Chinese question card when user input is needed.

Activate the copied global settings, preserving any existing Cannon versions first. These settings affect every Codex project under this account. Inspect existing Cannon settings and global instructions before replacement; reconcile any Cannon-specific customization, including existing project trust entries, into the staged files first:

```bash
mkdir -p "$HOME/.codex"
chmod 700 "$HOME/.codex"
if [ -e "$HOME/.codex/config.toml" ]; then
  cp -p --no-clobber "$HOME/.codex/config.toml" \
    "$CODEX_TRANSFER/config.before-fasrc.toml"
fi
if [ -e "$HOME/.codex/AGENTS.md" ]; then
  cp -p --no-clobber "$HOME/.codex/AGENTS.md" \
    "$CODEX_TRANSFER/AGENTS.before-fasrc.md"
fi
cp "$CODEX_TRANSFER/config.fasrc.toml" "$HOME/.codex/config.toml"
cp "$CODEX_TRANSFER/AGENTS.lxplus.md" "$HOME/.codex/AGENTS.md"
chmod 600 "$HOME/.codex/config.toml" "$HOME/.codex/AGENTS.md"
codex login --device-auth
codex login status
```

Use your browser on your own computer to complete device login; enable device-code login in your account settings if needed. [Official headless authentication guidance](https://learn.chatgpt.com/docs/auth)

The source config currently chooses `gpt-6.1-sol` with `high` reasoning, `danger-full-access`, and `approval_policy="never"`. Preserve the user's settings; these permission settings do not waive the campaign's human confirmation rules. If a model/setting is unavailable, use a Chinese question card before changing it. Retain `$CODEX_TRANSFER` as the private setup record and backup. The user-run account-wide Codex installation/configuration above is a narrowly authorized setup action; the running analysis agent's write scope remains the campaign's approved locations.

### A4. Repository pixi installation: known CUDA failure, repair delegated to the new agent

Reuse the existing pixi executable and shared cache established by the other handoff. This repository needs its own environment; the lep_calib environment is a different software stack.

**User-reported state (8 Oct 2026):** `pixi install` on `holy8a24301` failed to solve `default` for `linux-64`:

```text
py-xgboost-gpu >=2.1.4,<4 cannot be installed because there are no viable options:
py-xgboost-gpu 2.1.4 would require __cuda *, for which no candidates were found.
```

The preparation agent must only record this issue; Yiyang explicitly delegates its repair to the new Cannon agent. Preserve `pixi.toml` and existing environments during this document update. Continue with A5 and start Codex in A6 even while the analysis environment is unresolved; Codex and the separate VOMS installation below work independently of this workspace's environment.

The blocks below document the original installation attempt. Repeat installation after the new agent has addressed the root cause: the current manifest puts `py-xgboost-gpu` in the base dependencies, while the declared CUDA system requirement belongs to the separate `gpu` feature. Inventory the actual allocated GPU/driver and pixi version; establish a coherent CPU/default versus GPU environment design, preserve GPU training, and verify installation and real GPU execution on suitable Slurm workers. Record the resolved dependencies and any manifest/lockfile change. An assumed CUDA version or a silent CPU fallback does not establish working GPU training. [Pixi system requirements](https://pixi.prefix.dev/latest/workspace/system_requirements/)

```bash
source "$HOME/.bashrc"
ls -ld "$PIXI_CACHE_DIR"
spart
sshare -U
sinfo --version
scontrol show partition test
cd "$VVV_REPO"
salloc --account=iaifi_lab --partition=test --ntasks=1 \
  --cpus-per-task=4 --mem=16G --time=02:00:00
```

In the allocated compute-node shell, verify `hostname` and `SLURM_JOB_ID` before installing:

```bash
hostname
printf '%s\n' "$SLURM_JOB_ID"
source "$HOME/.bashrc"
cd "$VVV_REPO"
pixi install
git status --short
exit
```

The inspected repository has `pixi.toml` but no tracked `pixi.lock`. This install resolves the declared default environment and may create `pixi.lock`; retain it and record the resolved versions. If a lockfile exists in the actual clone, use its supported frozen/locked install mode to preserve it. Do not copy lep_calib's lockfile into this repository.

The manifest declares Python, NumPy, pandas, uproot, matplotlib, mplhep and GPU XGBoost. It does not establish a complete CMS stack: ROOT/C++ headers, correctionlib, CMS grid tools, scipy, zstandard and CombinedLimit availability need separate inventory. Some may be transitive or present in CMSSW; verify rather than installing speculatively. The `gpu` environment has a CUDA system requirement; install it only on an appropriate GPU allocation if it is actually needed. This block installs the repository's existing default dependencies, not new dependencies.

### A5. Set up the CMS certificate and proxy on Cannon

**User-reported state (8 Oct 2026):** Yiyang placed `myCertificate.p12` in `$HOME/.globus` on Cannon, extracted `usercert.pem` and a passphrase-protected `userkey.pem`, and set both PEM files to mode 400. Reuse these existing files. The repeated certificate extraction is already complete. Keep the P12 and PEM originals in `$HOME/.globus`; new proxy outputs go in private directories under `$HOME/.globus/proxies/`, shared across this account's projects.

Back on the Cannon login node:

```bash
cd "$HOME"
umask 077
export X509_USER_CERT="$HOME/.globus/usercert.pem"
export X509_USER_KEY="$HOME/.globus/userkey.pem"
ls -ld "$HOME/.globus"
ls -l "$X509_USER_CERT" "$X509_USER_KEY"
openssl x509 -in "$X509_USER_CERT" -noout -dates
command -v voms-proxy-init voms-proxy-info
```

Keep the existing mode 400 permissions and enter the PEM key's passphrase interactively. The certificate must correspond to the current CMS VO registration. Yiyang's `voms-proxy-init -voms cms` on `boslogin06` returned `command not found`; Branch 1 below is the installation path for that situation. The PEM credentials in `~/.globus` are the normal VOMS locations. [VOMS client credential guidance](https://italiangrid.github.io/voms/documentation/voms-clients-guide/3.0.2/)

**Branch 1: install the missing VOMS client on Cannon, configure CMS trust, then create a proxy.**

1. Install the conda-forge C/C++ VOMS client through the existing pixi executable. This is a separate account-wide tool environment, independent of `ScoutingVVVTools/pixi.toml`, and uses the existing global-tool PATH/cache. Yiyang requested VOMS installation; these are tutorial commands for Yiyang, not commands executed by the preparation agent. Install this one package and its solver-selected runtime dependencies, with no sudo or second package manager. The conda-forge recipe explicitly checks `voms-proxy-init` and `voms-proxy-info`. [VOMS package recipe](https://github.com/conda-forge/voms-feedstock/blob/main/recipe/meta.yaml), [pixi global install](https://pixi.prefix.dev/latest/reference/cli/pixi/global/install/).

```bash
source "$HOME/.bashrc"
cd "$HOME"
pixi global install --channel conda-forge --environment vvv-voms \
  --expose voms-proxy-init --expose voms-proxy-info 'voms=2.1.3'
command -v voms-proxy-init voms-proxy-info
voms-proxy-init --version
voms-proxy-init --help
voms-proxy-info --help
```

If the installation fails, keep its exact output for the new agent. A successful client installation supplies executables; the CMS endpoint and CA/VOMS trust configuration still need the next step.

2. Copy the public grid configuration from lxplus into a new private snapshot under `$HOME/.grid-security/`. This is reusable account-wide configuration, shared across projects independently of their repositories. It contains CA certificates/CRLs, VOMS trust anchors and server endpoint definitions; the personal certificate and key stay in Cannon's `~/.globus`. First verify the remote locations, then run the transfer only when all three exist. If lxplus reports a different layout, resolve the actual paths before transferring.

```bash
export VVV_LXPLUS=yiyangz@lxplus.cern.ch
ssh "$VVV_LXPLUS" \
  'ls -ld /etc/grid-security/certificates /etc/grid-security/vomsdir /etc/vomses'
```

```bash
cd "$HOME"
umask 077
mkdir -p "$HOME/.grid-security"
chmod 700 "$HOME/.grid-security"
export CMS_GRID_SETUP=$(mktemp -d "$HOME/.grid-security/snapshot-XXXXXX")
rsync -aL --info=progress2 \
  "$VVV_LXPLUS:/etc/grid-security/certificates" \
  "$VVV_LXPLUS:/etc/grid-security/vomsdir" \
  "$VVV_LXPLUS:/etc/vomses" "$CMS_GRID_SETUP/"
ls -ld "$CMS_GRID_SETUP/certificates" "$CMS_GRID_SETUP/vomsdir/cms" \
  "$CMS_GRID_SETUP/vomses"
```

The sources have no trailing `/`, so their names are retained in the destination; `-L` copies symlink targets as real files, keeping the snapshot usable on Cannon. Verify that `vomses` contains CMS endpoints and `vomsdir/cms` contains their trust anchors. CAs and CRLs need periodic refresh; retain this snapshot and transfer a fresh one before switching paths when it becomes stale. This relocation follows the upstream VOMS trust configuration and the grid client's documented user-local trust-directory pattern. [VOMS trust configuration](https://github.com/italiangrid/voms-clients#configuring-trust-for-voms-servers), [user-local grid configuration](https://lhcb.github.io/starterkit-lessons/self-guided-lessons/local-grid-proxy.html).

3. Persist the selected account-wide certificate and trust paths in `~/.bashrc`. The existing PATH already exposes pixi's global executables. This block records the actual home-directory snapshot path from step 2, while the proxy filename remains a per-renewal choice. If an earlier `scouting-vvv CMS credentials` block points into a repository, retain it and append this account-wide block after it so the shared-home paths take precedence. If the account-wide marker below already exists, inspect it and reconcile its selected paths before proceeding; preserve the old configuration and snapshots.

```bash
if grep -q -F '# >>> CMS grid configuration (account-wide) >>>' "$HOME/.bashrc"; then
  sed -n '/^# >>> CMS grid configuration (account-wide) >>>$/,/^# <<< CMS grid configuration (account-wide) <<<$/p' "$HOME/.bashrc"
else
  cp -p --no-clobber "$HOME/.bashrc" \
    "$HOME/.bashrc.before-cms-$(date +%Y%m%dT%H%M%S)"
  cat >> "$HOME/.bashrc" <<EOF
# >>> CMS grid configuration (account-wide) >>>
export X509_USER_CERT="\$HOME/.globus/usercert.pem"
export X509_USER_KEY="\$HOME/.globus/userkey.pem"
export X509_CERT_DIR="$CMS_GRID_SETUP/certificates"
export X509_VOMS_DIR="$CMS_GRID_SETUP/vomsdir"
export CMS_VOMSES="$CMS_GRID_SETUP/vomses"
# <<< CMS grid configuration (account-wide) <<<
EOF
fi
source "$HOME/.bashrc"
```

4. Create a new proxy with the existing PEM files and the explicit CMS configuration. Create the parent directory first, and check directory creation before using its path. A failed `mktemp` inside `export VAR=$(...)` can leave an empty variable, producing `/x509up` as the output path. The conditional plain assignment below preserves the failure status; export the proxy path only after VOMS creation succeeds. Enter the PEM key's passphrase when prompted:

```bash
umask 077
unset X509_USER_PROXY
if mkdir -p "$HOME/.globus/proxies" &&
   chmod 700 "$HOME/.globus/proxies" &&
   CMS_PROXY_DIR=$(mktemp -d "$HOME/.globus/proxies/cms-XXXXXX"); then
  if voms-proxy-init --rfc --voms cms --valid 192:00 \
    --cert "$X509_USER_CERT" --key "$X509_USER_KEY" \
    --certdir "$X509_CERT_DIR" --vomses "$CMS_VOMSES" \
    --out "$CMS_PROXY_DIR/x509up"; then
    export X509_USER_PROXY="$CMS_PROXY_DIR/x509up"
    chmod 600 "$X509_USER_PROXY"
    voms-proxy-info --file "$X509_USER_PROXY" --timeleft
    voms-proxy-info --file "$X509_USER_PROXY" --vo
    voms-proxy-info --file "$X509_USER_PROXY" --actimeleft
    printf 'Proxy path: %s\n' "$X509_USER_PROXY"
  else
    printf 'Proxy creation failed; preserve the output for diagnosis.\n' >&2
  fi
else
  printf 'Proxy directory creation failed; resolve it before continuing.\n' >&2
fi
```

Require VO `cms` and positive proxy and VOMS attribute lifetimes. The requested 192 hours may be capped; the shorter effective lifetime determines renewal. A CA/CRL, endpoint or VOMS signature error needs its actual configuration fixed before continuing. Successful proxy creation is followed by worker DAS/XRootD validation by the new agent. [CMS proxy usage](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookXrootdService)

**Branch 2: if local VOMS setup remains blocked, hand the exact failure to the new agent.**

Preserve the PEM files, installation output and trust snapshot. Start Codex through A6 with the certificate and available trust paths even if a proxy has yet to be created. Record `X509_USER_PROXY` as unset; the agent must finish VOMS setup and create/validate a real CMS proxy before authenticated data access. Reuse the Cannon certificate's current identity and CMS membership. An lxplus proxy transfer would be a separate fallback requiring an explicit decision.

```bash
unset X509_USER_PROXY
```

The agent must verify CMS VOMS attributes, grid CA trust, DAS authentication and XRootD access on an allocated Cannon worker before conversion. Pass the selected proxy and CA/trust paths to each worker; read the long-lived key only where proxy creation/renewal requires it. Further software/container/trust-resource installations follow the campaign's confirmation rules. For renewal, create another private proxy directory and update the campaign's selected proxy path explicitly, preserving the prior proxy.

### A6. Save this session's environment, reuse lep_calib's tw, start Codex

After a successful A5, `X509_USER_PROXY` points to the actual new proxy. If A5 is blocked, leave it unset; the saved empty value below records that pending state. Codex can start while proxy setup and A4's CUDA repair remain pending. Stable certificate/trust paths and PATH come from `~/.bashrc`. Save the selected paths without printing secrets:

```bash
cd "$VVV_REPO"
if [ -e "$VVV_REPO/build/fasrc/session.env" ]; then
  cp -p --no-clobber "$VVV_REPO/build/fasrc/session.env" \
    "$VVV_REPO/build/fasrc/session.env.before-cms-$(date +%Y%m%dT%H%M%S)"
fi
cat > "$VVV_REPO/build/fasrc/session.env" <<EOF
source "\$HOME/.bashrc"
export X509_USER_CERT="\$HOME/.globus/usercert.pem"
export X509_USER_KEY="\$HOME/.globus/userkey.pem"
export X509_CERT_DIR="${X509_CERT_DIR:-}"
export X509_VOMS_DIR="${X509_VOMS_DIR:-}"
export CMS_VOMSES="${CMS_VOMSES:-}"
export X509_USER_PROXY="${X509_USER_PROXY:-}"
EOF
chmod 600 "$VVV_REPO/build/fasrc/session.env"
```

This block preserves any previous `session.env` before recording the new certificate/trust/proxy paths. After a renewal or a trust-directory refresh, update the session record the same way and reload it in the running tmux shell.

Reuse the existing `tw` function installed by the lep_calib handoff in `~/.bashrc`. It attaches to or creates the shared tmux session named `claude`, and records its login node in `~/host.txt`. Use that same session for Codex:

```bash
source "$HOME/.bashrc"
type tw
tw
```

Inside the attached `claude` session, use a shell window. If the current window is occupied by the lep_calib agent, press `Ctrl-b`, then `c` to open a shell window in this same session, preserving the running process. Load the campaign environment explicitly, because an existing tmux server may have older environment variables:

```bash
source /n/holystore01/LABS/iaifi_lab/Lab/yiyangz/CMS_Run3_VVV_Scouting/ScoutingVVVTools/build/fasrc/session.env
cd "$VVV_REPO"
codex
```

For later reconnection, log in to FASRC, read `~/host.txt`, connect to that recorded login node, then use the same `tw` command:

```bash
ssh "$USER@$(cat "$HOME/host.txt")"
```

In the shell on that node:

```bash
source "$HOME/.bashrc"
tw
```

The shared home stores `~/host.txt`; the tmux session itself lives on the recorded login node.

Give the new agent this startup instruction:

> Read tmp/prompt.md completely and execute Part B. Read the current AGENTS.md, README.md and historical records completely. Run the entire framework from registered source MC/data, producing fresh ntuples and all downstream results on FASRC Cannon. Use the copied Codex settings. Every matter requiring my input must be a visible Plan-style question card in Chinese, including confirmations and download/deletion permission; do not ask in ordinary messages. Verify the question-card feature works at the first required decision. You may adapt the server integration; ask before any physics fix. Preserve reusable ntuples in Lab. Keep working, including periodic job monitoring, until the task is complete, a question is awaiting my reply, or an unsolvable problem blocks progress.

## Part B — instructions for the new agent

### B1. Objective, scope, and authority

Actually complete a fresh analysis campaign on FASRC Cannon using the current checkout. Begin with the NanoAOD-style source datasets registered in the current `src/sample.json`; produce new analysis ntuples, then every dependent product. “Ntuple production” here means this framework's conversion of those registered inputs, as in the old prompt. The repository does not provide a complete ScoutingNano/CRAB upstream-production campaign. If Yiyang means regenerating the upstream /USER datasets too, ask through a Chinese Plan-style card before expanding scope or installing upstream tools.

Keep the current five mutually exclusive analysis trees:
`boosted3`, `boosted2_resolved`, `boosted2_unresolved`,
`boosted2_leptonic_1lep`, `boosted2_leptonic_2lep`.
They are analysis channels, not the classifier's process classes.

Completion requires fresh PU CSVs, ttbar-CR ntuples and JMS/JMR fit, analysis nominal ntuples and all eight configured MC JES/JER/JMS/JMR variations, an explicitly settled mixing/training flow, a model and diagnostics per tree, signal regions per tree, QCD ABCD outputs, all evaluable systematic inputs, Data/MC and class-shape plots, and full-systematics combine expected significance/limits. Include combined, signal-class, signal-sample and individual-channel scenarios, with MC-true and ABCD QCD outputs. Resolve the scope of standalone studies through cards rather than silently dropping them or using incompatible data.

Yiyang authorizes the normal campaign's compilation, Slurm submission, analysis execution, necessary server adaptations, monitoring, recovery and reruns. This overrides a generic requirement to ask again for each normal run. Read the complete relevant sources and data flow before adaptations, state the concrete plan, and document changes. Keep existing lxplus/Purdue/other-site support. This authorization excludes physics fixes or definitions, destructive actions, new dependencies/downloads, new input scopes and git mutations. Ask for those through a card before proceeding.

All campaign products must use a new namespace. Existing ntuples, trained models, fits, SR CSVs, systematic JSONs and combine outputs are historical, not fresh products. Source registry metadata is a starting point to verify, not proof of new processing.

### B2. Binding interaction, persistence, and safety rules

1. **Every request for human input uses a visible Plan-style question card in Chinese.** This includes clarifications, confirmations, approval of a plan, downloads/installations, deletion, conflicting instructions, problems, physics fixes and unavailable resources. Prefer the platform's synchronous Plan-style question mechanism; with Codex this is the visible question-card capability enabled by `default_mode_request_user_input=true`. Ordinary commentary/final messages may report information but must not ask Yiyang to reply. Never replace a required card with an invisible asynchronous card or a prose question.
2. Explain the evidence and impact in the card, give concrete options and a recommendation with a reason, and ask one round at a time. Wait for an explicit answer before dependent work. Restate and record the answer. Silence, elapsed time and empty answers are not permission. Do independent work while waiting when it remains possible.
3. Verify that the copied feature is active. If a tool is restricted to Plan mode, use the supported Plan-mode workflow. If the required UI cannot be made available, treat that as a real tooling blocker; report the limitation without putting a question in prose or using “default approval.” Do not promise an unsupported mode switch.
4. **Keep executing until the objective is achieved.** Before completion, stop only while a required card awaits an answer, or when an unsolvable problem blocks all useful progress. A submitted job, a report, a long queue wait or a long-running job is not a stopping point. Continue periodic job checks and recovery; do not hand monitoring back to Yiyang merely because jobs take hours. This expressly supersedes the lep_calib reference's “stop while long jobs run” exception and old handoff stop instructions.
5. Monitor at a sensible interval, initially about 2–5 minutes, with lightweight scheduler queries and bounded waits that keep the session responsive. If the available waiting tool caps individual waits, use shorter waits between scheduled checks. Repeatedly query accounting only when state changes or diagnostics require it; avoid tight polling. During long intervals, continue useful independent work. Keep concise Chinese status updates and timestamped monitoring records.
6. Preserve all files and user edits. **Every deletion requires Yiyang's prior explicit confirmation through a Chinese Plan-style card**, including intermediate files, caches, stale sidecars, failed outputs and program-driven cleanup. Audit cleanup calls and replacement behavior before executing existing programs. Prefer new paths and retained files. A previously copied allow-rule, “temporary” label or generic execution approval does not authorize deletion.
7. Scope writes to this repository plus the approved project siblings `ntuples/` and `results/`, approved project scratch, and the already configured pixi cache `$IAIFI/.cache/pixi`. Scratch is for intermediates. Preserve reusable ntuples, final products and recovery state in Lab. These specific user grants override a blanket repo-only instruction for these paths; other users' directories and other projects remain read-only. Further write locations require a card.
8. Part A's user-run permanent shell setup, account-wide Codex installation/global configuration, VOMS setup and one-time clone are narrow setup actions, not permission for the agent to modify arbitrary home settings, install arbitrary software or mutate git. Reuse the existing Cannon P12/PEM originals in `~/.globus`, with the PEM files at mode 400, and the reusable account-wide CA/VOMS configuration under `~/.grid-security/`; preserve them and use private proxy directories under `~/.globus/proxies/`. The campaign records the selected shared-home proxy path in its session environment. Codex is shared across this FASRC account's projects; the analysis environment and campaign outputs retain their project scope. Git after setup is read-only. Do not commit, push, fetch, switch branches, stash, reset, checkout or rebase. Ask before any `.gitignore` change and report every new file's tracked/untracked/ignored status.
9. Reuse software already installed for the other FASRC project when compatible. A4's `default` environment installation failed on missing `__cuda`; Yiyang delegates the manifest/environment repair to this new agent, with real worker validation and GPU training preserved. Installing the repository's existing dependencies remains scoped to its manifest. Yiyang also requests the separate VOMS client installation specified in A5; record whether it has been completed and finish that exact installation if still pending. Part A's public grid-configuration transfer is a user-run setup step. Every other dependency, external calibration/resource/container download or new installation needs a card describing exact source, version, destination, command and purpose. No unrestricted install permission is inherited from the other project's handoff.
10. No local tests. Do not run pipeline code, compile probes, import-check scripts or heavy verification on a login node. Authorized compilation and campaign verification run on Slurm workers. Follow the repository's testing ban; if additional tests would be necessary, resolve the scope through a card. Read-only shell/file/git/documentation inspection is permitted.
11. Credentials remain private. Never print key/proxy/auth contents, include them in job manifests, commit them or copy them into results. Use explicit proxy paths, protected permissions and actual expiry checks; export only the needed credentials into each worker. A transfer retains expiry. Ask through a card for renewal when needed.
12. Read `AGENTS.md` and `README.md` completely. Follow style and tool-layout rules. Code comments and documentation stay English; user conversation/cards stay Chinese. Physics/source assumptions require evidence from current code, real inputs and official documentation.
13. Store new `REQUIREMENTS.md`, `PLAN.md`, `LOG.md` and `progress.json` under `tmp/fasrc_<campaign>/`, in English. Transcribe these rules and confirmed user decisions, append every new requirement/answer, record exact commands/job IDs/software/input/config hashes, and preserve history. Take timestamps from the clock, including timezone. Old CERN timestamps are not Cannon measurements.
14. At every small-stage checkpoint, reread the entire new plan, log, progress, requirements and this prompt, plus the applicable repository instructions, word by word. Complete truncated reads. Update the records and send an informational Chinese status message; then continue. Synchronize new requirements into AGENTS.md and keep README.md consistent with code changes as required by the repository. The prompt-only preparation did not modify those tracked files.
15. Fix root causes with the smallest coherent change. Preserve existing deleted/disabled code unless Yiyang confirms restoration. Report unrelated problems without fixing them. **A physics bug always requires a card and confirmation before editing**, even when the proposed answer looks obvious or a historical prompt said “bug fixes authorized.”
16. Every systematic uncertainty that can be evaluated goes into combine. An enabled nuisance with missing input is an error. Never conceal missing weights, failed fits, incomplete batches or unavailable corrections using fabricated unity, zero or “success” records.

### B3. Confirmed physics decisions and historical issues

Use the current code to determine whether a historical issue has already been resolved. Preserve current intentional improvements; do not apply the old repair list mechanically.

- **Data:** only `2024G_UParT_v6` is authorized for this campaign. Current `sample.json` also contains `data_2024`, with broader G/H/I inputs; its presence does not expand authorization. Explicitly select the permitted data, since mode-0's fallback is MC only.
- **Luminosity:** expected sensitivity remains 255.8 fb^-1. The registered G-v6 data has 27.51 fb^-1 and the historical user decision projects the Data/MC display to 255.8. A projection scales the data uncertainties consistently and must be labelled; it does not create more observed statistics. Reconcile with the current plotting implementation. The historical question about extending that projection to the JMS/JMR control fit was unanswered. Keep the actual CR statistical precision; use a card to settle any affected fit/display change.
- **Pileup:** retain the user's original 2024 C–I profile and regenerate MC weights. The current repository recommends profiles matching processed lumis in some places; do not silently override the user decision. If current implementation or data makes the retained choice inconsistent, explain the evidence through a card before changing it.
- **MC membership:** all registered MC is available, but overlapping inclusive/exclusive samples must not be blindly summed. Follow the current class_groups as the baseline, check source identity and physical overlaps, and ask before membership/category changes.
- **Generator normalization:** current conversion writes real `genWeight` for all MC, and the registry already has `genweight_mean` plus PU means. This does not by itself prove every downstream weight builder uses them. The historical instruction was to inspect Runs, signed weights and upstream filtering before proposing any coherent full-chain physics change. Review current training, SR, QCD, plotting, fitting and syst weighting separately. Resolve actual remaining physics discrepancies through a card; do not replace the current weighting solely because the old prompt described another version.
- **DY:** the old duplicate `dytoll_ht400to800_mll120` is absent in the inspected registry. Both current training and CR reference `dy_h400to800_m120`, whose current xsection is 175.3. The alias-selection repair is therefore obsolete. Retain the current canonical entry; verify provenance if still necessary, rather than recreating the removed alias or automatically reopening a settled fix. Old McM evidence did not validate either historical value and must not supply a numerical correction on its own.
- **Resolved AK4 cleaning:** the historical confirmed intent is cleaning against the two retained AK8 jets. Current `ak8_kept2` still has a general pT/eta selection and pT sort, with no obvious two-object cap in that JSON. Inspect the complete engine/collection semantics before claiming a remaining discrepancy. If the discrepancy remains, explain the current behavior, object identity/order and variation effects through a card, and obtain confirmation before editing this updated version's physics.
- **Models:** the inspected default remains BDT, five trees, six process classes (VVV, VH, Top, VV, VJets, QCD), with unmixed input and a Purdue absolute path. Preserve model/class choices; resolve the mixing flow through a card if it is not already settled in current source/records. Mix and jet-variation processing must preserve the required sample split/event correspondence.
- **Signal regions:** prioritize the final combine sensitivity/expected significance with all systematics, as previously confirmed. If a full search is too costly, a transparent approximation or staged reevaluation is allowed; state cost/stopping criteria and ask about unsettled choices. Compare existing rectangle/histogram candidates fairly on the same data/model/constraints and use full-systematics evaluation of promising solutions. A completed finite-candidate search proves only that finite pool's optimum; do not claim a continuous global optimum.
- **Standalone studies:** old b-veto, trigger-efficiency, ABCD-2D and likelihood-scan diagnostics can have incompatible sources/definitions. Survey them and use a card to settle scope/data/definitions. Do not silently use the old F/ZeroBias or UParT-v2 inputs, remove them from “full workflow,” or report an analytic approximation as a computed likelihood scan.

The three already-authorized PU reference files transferred by A2 are external calibration inputs, not reused campaign outputs. Verify these hashes:

| File suffix | Bytes | SHA256 |
| --- | ---: | --- |
| 66000ub.root | 4470 | 02be96ef021370cb8c6e6d66e49bc40bf3e9e9c75711b7d7491c70d7c2f97454 |
| 69200ub.root | 4466 | 5e379a2004788f693187f6a579a81f2530a27a00482e39b559337d026c40fb93 |
| 72400ub.root | 4468 | 3b85a78a4a1ee929edccf9b785e5c50095022cdd74e5a29539fba4842dfcb253 |

Full filenames are `dataPileupHistogram-2024CDEFGHI_Golden-<suffix>`, under `tmp/lxplus_full_20261001/inputs/pileup/`. Source: the three exact official links in README.md, under `https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions24/PileUp/`. Their histogram semantics still need worker validation. Preserve existing copies; missing/changed files require investigation, and a new download uses a card.

### B4. Current-version audit: resolved items and remaining integration hazards

These are source observations at the recorded commit, not claims of Cannon execution or a complete new source audit. Recheck against the actual clone and read whole files before implementation.

| Historical concern | Current state and action |
| --- | --- |
| Shifting DAS lists, weak resume provenance | Converter already snapshots sorted input lists in `.files`, checks input/config/binary/calibration hashes in `.meta`, and validates sidecars/trees. Preserve this; do not reimplement the old repair. |
| Slow entry-by-entry merge | Converter already performs fast basket-copy merge and validates entry counts. Retain it. |
| Concurrent sample.json updates | Converter has an exclusive flock and per-process temporary writer. Preserve it and consistently use the campaign's sample metadata path. |
| Missing generator output / duplicate genWeight | All-MC binding/output and duplicate-booking guard exist. Verify schemas; do not blindly repeat the old patch. |
| JEC integration into rewritten engine | The current port is already merged. Keep the slot-based engine, all configured jet corrections, Type-1 MET and eight variation trees. |
| Superseded DY alias | One canonical entry remains; old duplicate-selection work is obsolete. |
| combine cleanup default | Current `keep_work` is already true. Preserve it; audit other automatic cleanup separately. |
| Slurm proxy copying | `run.py.main()` still ignores the supplied proxy path when preparing `--slurm`: it reads `/tmp/x509up_u<uid>` and writes `/depot/cms/users/<user>/...`. Cannon needs a correctly shared selected credential, not that Purdue path. |
| Pixi default environment / CUDA | User's 8 Oct installation on `holy8a24301` failed because the base GPU XGBoost dependency requires `__cuda`. Repair is delegated to the new agent (A4); this preparation leaves `pixi.toml` unchanged. |
| Slurm mode coverage | `--slurm` dispatches sample jobs for modes 0/1/6. Python modes run immediately in the invoking process, and combine mode 7 also runs immediately. Wrapping them in actual sbatch jobs is required; a flag alone does not allocate resources. |
| Compilation / filesystem safety | `run.py` compiles before submitting sample jobs, uses OpenMP tempfile probes and removes them, and local C++ modes compile/remove the plain binary. Run compilation on workers, retain uniquely named build artifacts under `build/`, and preserve the tracked `selections/convert/convert_branch`. Audit converter temp/sidecar cleanup before running it. |
| Old cluster defaults | Default account is `cms-express`; training points to `/depot/...`; several downstream syst configs still point to `*_hist` SR paths. Replace execution paths coherently through campaign configs/server integration. |
| ROOT metadata format | Current `qcd_est.py.write_root_output` assigns dictionaries at `metadata/signal_regions` and `metadata/abcd_closure`. Current uproot >=5.7 uses RNTuple for that operation, while combine expects TTree. Verify and resolve producer/consumer compatibility as a format fix, preserving numbers. |
| Documentation contradictions | AGENTS.md still says the entire `jet_pt_correction` block is rejected, but the current converter loads it and current configs use it. Only specific obsolete per-variation keys are rejected. Follow source/current schemas and update directly affected documentation when adapting code. |
| Physics-weight consistency / AK4 cleaning | Remain investigation items, not approved code changes. Use B3 and question cards. |

Uproot source for metadata behavior: [official writing-TTrees documentation](https://uproot.readthedocs.io/en/stable/basic.html#writing-ttrees-to-a-file). Confirm against the installed worker version. A format-only correction is a normal adaptation; any change to physics yields/statistics needs separate confirmation.

For campaign configs, **use absolute paths** for all cross-stage inputs/outputs and record each program's path-resolution base. The C++ converter loads `branch.json` and `selection.json` beside its active config, so an isolated CR or analysis config needs the corresponding complete sibling files. Several Python scripts resolve relative inputs against their script directory rather than the custom config's directory. Saved BDT config copies are later read by different scripts. Moving only config.json without its dependencies or assuming all relative paths use the same base will break the workflow.

Keep fresh sample metadata separate from historical registry counters when appropriate, and ensure every downstream copy points to the same campaign metadata. Retain the current source/cross-section identities and ask before physics edits. Preserve `.files`, `.meta`, `.raw_entries`, `.lumis`, processed-lumi JSON and all data needed to validate staged outputs.

### B5. FASRC execution policy: inspect, then adapt

Cannon uses Slurm. Use sbatch/squeue/sacct/scontrol rather than adding CERN eossubmit or assuming an HTCondor service. Preserve other-site support; implement a Condor path here only if site evidence and a confirmed requirement actually warrant it. [FASRC running jobs](https://docs.rc.fas.harvard.edu/kb/running-jobs/)

Read the transferred `build/fasrc/reference/IAIFI_Computing_Cannon-Resources-for-Investigators.pdf`, especially storage, account and GPU sections. It is dated 2024-05-20: `iaifi_gpu` and `--account=iaifi_lab` are useful starting points, while current access, names, hardware, limits and quotas require inspection. `iaifi_gpu_priority` requires a separate grant; do not assume access or guaranteed immediate scheduling.

On the login node, lightweight read-only inventory can include:

```bash
hostname -f
id
spart
sinfo --version
sinfo -o '%P %a %l %D %G'
sshare -U
scontrol show partition shared
scontrol show partition sapphire
scontrol show partition iaifi_gpu
ls -ld /n/netscratch/iaifi_lab/Lab /n/netscratch/iaifi_lab/Lab/yiyangz
command -v sbatch squeue sacct root-config scram dasgoclient singularity apptainer
ls -ld /cvmfs/cms.cern.ch
```

Record failures as missing capabilities; `command -v` or a mount existing does not prove worker compatibility. Do not assume the CMS CVMFS repository, CMSSW release, grid trust store, outbound CMS network access or a usable container is available on all nodes.

- The C++ conversion is designed for CMSSW_16_1_0_pre4 with CMSSW's correctionlib (the documented working architecture is el9_amd64_gcc13). Inspect compatible central installations on workers first. Keep CMSSW and pixi activation separate; record compiler, ROOT ABI and correctionlib paths. Do not run the converter with an accidentally substituted incompatible user-site package.
- Python analysis uses this project's pixi or a suitable existing CMSSW environment. Inventory actual imports on an allocated worker, including zstandard decoding of ZSTD-5 ROOT output, scipy for the CR fit, uproot and the installed GPU XGBoost. The other project's environment is not automatically interchangeable.
- Combine requires a suitable environment with built HiggsAnalysis/CombinedLimit, `combine` and `combineCards.py`. A pixi executable or ROOT installation alone is insufficient. Keep the repository's environment design. If a different environment/container would be needed, present the exact proposal and get approval.
- If CMS software is missing, investigate the site's existing SingularityCE/Apptainer capability and available compatible images. Do not presume which command is installed, pull an unapproved image, bind nonexistent CVMFS, or suggest a conda substitute as an already approved CMS environment. [FASRC container documentation](https://docs.rc.fas.harvard.edu/kb/singularity-on-the-cluster/)
- GPU training should use an accessible IAIFI GPU partition with an explicit GPU request, such as `--gres=gpu:1` after site inspection. Verify the allocated device/driver, XGBoost version and actual training device. A broad CUDA exception followed by CPU fallback is not successful GPU verification; ask if a real fallback is required.
- Use one task with `--cpus-per-task` for a single OpenMP/Python process, explicit memory and wall time, and `--account=iaifi_lab`. Match OMP/BLAS/XGBoost/process-pool threads to the allocation rather than the host's total cores.
- Use job arrays or bounded batches where appropriate, with an explicit concurrency throttle. Check current array/job limits and avoid floods of DAS requests. Use test partitions for short setup/verification, and suitable production partitions for actual full processing.
- Build a shallow stage dependency plan. Preserve the converter's afterany merge plus strict input validation if using its launcher; changing a dependency must preserve missing-batch detection. Later stages need verified upstream success. Account for failed dependencies and array tasks individually.
- Requeue queues are preemptible: use them only after confirming resumability and output safety. Do not depend on uncheckpointed multi-hour GPU training surviving preemption.

Login nodes are for Codex/tmux, read-only inspection, submission and small external transfers. Heavy computation, builds and large I/O must use allocations; tmux provides persistence, not compute resources. Use scratch for heavy I/O, retaining reusable ntuples in Lab. [FASRC quickstart](https://docs.rc.fas.harvard.edu/kb/quickstart-guide/)

Useful monitoring commands (replace JOB_ID with a recorded ID):

```bash
squeue -u "$USER" -o '%.18i %.28j %.12T %.10M %.9l %.6D %R'
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocCPUS -P
scontrol show job JOB_ID
jobstats JOB_ID
```

Check pending reasons, failed/OOM/TIMEOUT/PREEMPTED tasks, worker logs, input coverage and output completeness. The queue becoming empty is not proof of success. [FASRC convenient Slurm commands](https://docs.rc.fas.harvard.edu/kb/convenient-slurm-commands/)

### B6. Read the current complete pipeline before designing changes

Read this prompt, current AGENTS.md/README.md, all three historical records, the backup prompt, the IAIFI guide and relevant FASRC official docs. Treat the other project's handoff as a site/setup reference; its git pushes, extra installs, unrelated data access and stop-while-waiting rule do not apply here.

Read complete current sources/configs for each stage and their shared dependencies, including:

- `run.py`, legacy `run.sh`, `pixi.toml`, actual lockfile if present, `.gitignore`, `src/sample.json` and `src/simple_json.h`.
- `selections/weight/weight.C`, its config and `make_data_pileup.sh`.
- `selections/convert/convert_branch.C`, complete `branch.json` and `selection.json`, configs and `check_conversion.py`; corresponding CR configs/branches/selections.
- `selections/jms_jmr/jms_jmr_fit.py` and config, `plotting/config_ttbar_cr.json`.
- `selections/mix/mix.C` and config.
- `selections/BDT/train.py`, `model_io.py`, branch/selection/config files.
- Both `selections/signal_region/signal_region.py` and `signal_region_hist.py`, `openmp_region_select.cpp` and selected configs.
- `background_estimation/qcd_est.py` and config.
- Theory/PU syst sources, all four jet-syst wrappers, `selections/jet_syst_common/jet_variation_common.py`, merge_shards scripts and configs.
- `plotting/data_mc.py`, `class_shapes.py`, branch/config files.
- `combine/combine.C` and config, standalone studies and other variants that the chosen campaign actually uses.

Record reading coverage honestly. This handoff update inspected the dispatcher completely, current docs/configs, historical records and relevant source implementations, not every line of the large analysis sources. Earlier agent coverage predates the merged changes and is not a substitute for the new audit.

### B7. Execution plan and acceptance checkpoints

1. **Inventory and decisions.** Record the exact checkout/diff, software on login and workers, proxy expiry/VO, storage capacity, account/partitions and network access. Complete the delegated A4 CUDA/environment repair and any pending A5 VOMS setup, using the existing `~/.globus` PEM files and the selected trust/proxy paths. Transcribe requirements. Use Chinese cards for any remaining input, physics or installation decision. Where AGENTS.md requires remote-access confirmation, present the concrete registered DAS/XRootD source scope once and wait; that approval then covers the agreed campaign reads rather than asking again for every file.
   **Accept:** real worker CMS/data/software capability, clear sample scope and source manifest, working question UI, no guessed paths or dependencies.
2. **Adapt the server layer and establish a campaign.** State files/functions/behavior/config/output/downstream/docs changes. Implement the authorized Slurm/proxy/build/path/resource/format integration coherently, retaining existing-site support. Resolve cleanup before execution through retained paths or confirmed deletion scope. Use campaign-specific complete configs, absolute references and fresh Lab/scratch output paths. Synchronize requirements in AGENTS.md and code changes in README.md. Any physics issue goes to a card before editing.
   **Accept:** reviewable minimal diff, protected tracked binary/user edits, distinct build/config namespaces, reproducible worker environment and job manifest.
3. **Verify sources and metadata.** On workers, inspect schemas, Events/Runs/LuminosityBlocks, theory array layouts/counts and sample normalization/filters as required by current code. Use the current canonical samples. Obtain fresh processing bookkeeping without overwriting historical provenance. Ask about actual remaining physics inconsistencies.
   **Accept:** complete input list, explicit usable/missing inputs and justified normalization; no arbitrary repair from old notes.
4. **PU and corrected ttbar CR.** Validate the retained C–I reference histograms, rebuild PU CSVs, convert all required CR MC and only authorized G-v6 data, and fit new JMS/JMR with actual CR statistical precision. The fitter must read all newly produced CR chunks and required branches; check its current single-base-filename assumptions and missing-sample behavior. Review fit quality and errors.
   **Accept:** required CR samples/files complete, valid fit/diagnostics and fresh results JSON, independently traceable PU files.
5. **Analysis conversion and durable ntuples.** Convert agreed MC/data with the current nominal correction chain, new PU/JMS/JMR inputs and all eight MC jet variations. Use fixed file-list snapshots; retries must preserve source mapping. Validate every batch and merge with hashes/sidecars and entry totals. Report skipped MC files/truncation/read errors; data losses block completion. Transfer validated reusable ntuples and associated recovery/provenance data to `$VVV_PROJECT/ntuples/<campaign>/`. Preserve stage state and frozen source/binary/config identities across transfers; note that rebuilding/moving binaries or editing calibration/config hashes can invalidate resume metadata.
   **Accept:** full agreed coverage, all required nominal/variation trees and branch schemas (5 nominal plus 40 variations for MC), consistent bookkeeping, durable Lab copies.
6. **Mix and train.** Execute the confirmed mixing/training flow per tree using allocated resources and verified GPU support where required. Preserve memory-conscious training patterns. All saved models, copied configs and test split/reference files refer to this campaign. Account for zero-entry sample/channel behavior transparently.
   **Accept:** model per tree, required branch/loss/ROC/score/decorrelation/importance diagnostics, accurate split correspondence and successful actually executed reference checks.
7. **SR and QCD.** Compare appropriate optimizers under identical conditions; use a feasible full-systematics reevaluation strategy for the final choice. Produce disjoint SR CSVs for all five trees and corresponding ABCD outputs; inspect control-region populations, closure and the ROOT metadata types/numbers.
   **Accept:** one identified final SR definition per tree, stated optimum/certificate scope, correct references and non-stale downstream paths.
8. **Systematics and plots.** Run modes 8/9/11/12/13/14 on new inputs/models/SRs; merge all shards and verify sample/tree/SR coverage and physically valid ratios. Include applicable theory PDF/scale/ISR/FSR, PU, JES/JER/JMS/JMR and the current flat/ABCD/statistical nuisances. Determine additional evaluable systematics from real calibration inputs/definitions; ask when availability or physics is uncertain. Produce fresh Data/MC/class-shape plots with the confirmed display normalization.
   **Accept:** all enabled/evaluable nuisance inputs accounted for, no silent unity substitution, complete requested plots and consistency with QCD A-region selection.
9. **Combine and standalone scope.** Run the wrapper on proper compute workers with built CombinedLimit, five channels, current nuisances and `keep_work=true`. Inspect datacards, shapes, fit output/expected quantiles and correlations. Classify any current 0/inf placeholder row as zero-sensitivity or failure using evidence rather than reporting every row as a successful fit. Complete the agreed standalone studies.
   **Accept:** actual expected-significance/limit outputs for agreed scenarios, all failed/missing-fit cases resolved or explicitly blocked by a confirmed issue, retained work and fresh result provenance.
10. **Final verification and report.** Complete the full-record reread. Reconcile jobs, source/batch coverage, Lab ntuples/results and all cross-stage references. Report real output paths, software/config/input provenance, validation evidence, unresolved limitations and new-file git/ignore status. Finish only after the objective is achieved, or when a required card/unsolvable blocker genuinely prevents further progress.

Current launch mapping, to preserve when creating proper worker commands:

| Stage | Entry point |
| --- | --- |
| PU | mode 1, WEIGHT_CONFIG_PATH |
| CR / analysis conversion | mode 0, CONVERT_CONFIG_PATH |
| JMS/JMR fit | selections/jms_jmr/jms_jmr_fit.py --config <absolute-config> |
| Mix | mode 6, MIX_CONFIG_PATH |
| BDT | mode 2, BDT_CONFIG_PATH |
| General SR | mode 3, SCAN_CONFIG_PATH |
| Histogram SR | selections/signal_region/signal_region_hist.py, verify its current config interface |
| QCD ABCD | mode 5, QCD_EST_CONFIG_PATH |
| Theory / PU syst | modes 8 / 9, THEORY_CONFIG_PATH / PILEUP_SYST_CONFIG_PATH |
| JES / JER / JMS / JMR syst | modes 11 / 12 / 13 / 14, respective *_SYST_CONFIG_PATH |
| Data/MC / class shapes | modes 4 / 10, PLOT_CONFIG_PATH |
| Combine wrapper | mode 7, COMBINE_CONFIG_PATH |

Do not present an unadapted `python run.py ... --slurm` line as a working Cannon command. First make the appropriate server integration concrete, prove it on a worker, then submit the complete production campaign and keep monitoring it.

### B8. Initial historical records and ongoing progress

The copied lxplus plan/log/progress describe the 2026-10-01 campaign in another checkout. They contain the original answers, authorized PU download provenance and DY metadata research. They do not establish Cannon environment readiness, the current code's correctness, a current proxy lifetime, or new result completion.

At this handoff, no Cannon job, fresh Cannon ntuple or downstream result has been verified. Begin a new FASRC campaign record; retain historical records unchanged. Already resolved source issues are removed from the active repair plan, actual remaining issues are investigated, and every new physics correction is confirmed through a Chinese Plan-style question card.

The task is complete fresh production and analysis. Keep working through adaptation, execution, periodic monitoring, recovery and validation; ask visibly whenever human input is needed.
