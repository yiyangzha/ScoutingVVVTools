# lxplus full-workflow campaign plan

## Objective and completion criteria

Start from the MC and data sources registered in `src/sample.json`, produce fresh ntuples, and run the complete analysis with the current systematic uncertainties on lxplus. All registered MC samples are available. The only authorized data sample is `2024G_UParT_v6`.

Completion requires fresh pileup weights, the corrected ttbar control region and a fresh JMS/JMR fit, nominal and JES/JER/JMS/JMR variation ntuples, the chosen mixing and training flow, signal regions for all five analysis trees, QCD background outputs, all systematic ratio inputs, Data/MC plots, and the configured combine significance and expected-limit outputs. Record and resolve failed or held jobs and validate completeness and input/output consistency. Additional standalone studies must have their scope confirmed when their inputs or physics assumptions differ from the campaign.

Preserve old outputs. Every downstream input must reference products from this campaign. A dependency that is an external calibration input must be obtained or generated using a confirmed source and period.

## Requirements and authority

- The user authorizes execution, compilation, job submission, reruns, straightforward cluster adaptations, and bug fixes within this campaign. Normal execution needs no repeated permission request.
- Read the complete pipeline and data flow before implementation. The source audit is still in progress; do not claim that every repository file has already been read.
- Keep existing cluster support and add lxplus support through the CERN eossubmit module.
- Use all available MC as appropriate to the confirmed process groups. Use only `2024G_UParT_v6` data. Resolve aliases and overlapping physical samples before combining yields.
- Use the backed-up grid proxy. The backup preserves the credential contents and the original expiration time.
- Investigate batch GPU resources before asking for an alternative.
- Ask about unclear, uncertain, strange, contradictory, or unmet requirements. Changes to functionality, physics definitions, or category definitions require confirmation.
- Do not install or download dependencies or external resources without explicit permission for the action. Confirm remote DAS/XRootD access as required by AGENTS.md.
- Keep writes, temporary files, environments, and caches inside this repository. Preserve existing files and user edits. Use read-only git commands. Run no local tests.
- Keep plans, logs, and progress under `./tmp/`, ignored by git. Update AGENTS.md with every new user requirement; keep README.md synchronized with code changes.
- After each small stage, reread the entire log, plan, progress record, and AGENTS.md word for word, including earlier contents.
- Use Plan-mode question cards where available. If unavailable, present clear questions and options together, then stop execution until the user answers. Do not rely on asynchronous cards the user cannot see.

## Execution stages

1. Complete the source/configuration audit and resolve the first physics/input questions. Verify: an explicit source coverage record and confirmed luminosity, pileup source, and DY alias policy.
2. Record the exact implementation plan and software/input sources. Reuse installed software where possible; request permission for specific missing downloads. Verify: CMSSW/correctionlib, Python dependencies, combine, eossubmit, credential validity, GPU matching, and repository-only paths.
3. Implement the required lxplus submission and file-safety adaptations and confirmed fixes. Candidate files/functions: `run.py` compilation/dispatch/proxy/log handling; conversion completeness and theory-branch handling in `selections/convert/convert_branch.C`; QA paths in `selections/convert/check_conversion.py`; explicit TTree metadata writing in `background_estimation/qcd_est.py`; campaign-specific copies of the stage configurations; AGENTS.md and README.md. Final edits depend on the completed audit and user decisions. Preserve legacy cluster behavior. Verify by read-only diff review and checks on batch workers, with no local tests.
4. Establish a fresh output namespace and job manifest under an ignored repository path. Use explicit current input lists, unique logs, saved software/configuration provenance, and dependencies that require upstream completeness. Serialize updates to shared sample metadata. Verify: no old products are selected as campaign inputs and no tracked binary is overwritten or deleted.
5. Run pileup production from MC and the confirmed data profile; convert the ttbar control region and fit fresh JMS/JMR values. Verify: all required inputs and samples present, fit diagnostics reviewed, normalization consistent, and no stale calibration reused.
6. Convert the analysis samples with nominal plus eight jet variations. Verify: complete source-file/batch coverage, branch schema, nominal/variation tree presence, merged entry counts, normalization metadata, and no silent skipped files.
7. Run the confirmed mixing/training flow for all five trees on suitable batch resources. Verify: model outputs, exact test segments, copied configuration provenance, reference predictions, and training/decorrelation diagnostics.
8. Optimize signal regions for all five trees with the confirmed optimizer and compute the background estimate. Verify: valid disjoint regions, matching prediction references, usable ABCD control regions, and explicit TTree metadata consumed by combine.
9. Produce theory, pileup, JES, JER, JMS, and JMR inputs using the new models/regions/ntuples, plus Data/MC and class-shape plots. Verify: nuisance coverage for every relevant sample/tree/region, finite valid ratios, consistent sample groups and normalization, and complete requested plots.
10. Run combine with all evaluable enabled nuisances. Verify: datacards, fit diagnostics, significance/limit outputs for configured combined/class/sample/channel scenarios, and explicit reporting of missing quantiles or failed fits rather than treating them as successful measurements.
11. Perform the final full-record review and report actual completed outputs, validation evidence, unresolved issues, and job identifiers/status. Verify git status and ignore rules for every new file.

## First decision round: answers received

1. Luminosity: retain 255.8 fb^-1 for expected sensitivity. The user specifies the listed v6 G data with original luminosity 27.51 fb^-1, normalized to the 255.8 target. An optional follow-up card asked whether this display projection should also cover the JMS/JMR control-region fit; it returned no answers. Continue independent work and keep the fit/display implementation scope explicit before changes.
2. Pileup inputs: retain the original 2024 C-I data profile. The user explicitly permits obtaining the three official ROOT reference files linked in README.md. Store these external calibration inputs inside ./tmp and regenerate every MC pileup weight from the registered MC sources.
3. DY alias and cross section: verify the cross section first. The same dataset is registered under `dytoll_ht400to800_mll120` (56.84) and `dy_h400to800_m120` (175.3). Research the source; neither value has been selected or approved yet.

## Other audit findings requiring classification or confirmation

- Default training input_root is a Purdue `/depot/...` path. Default SR production and downstream `*_hist` CSV paths disagree. Confirm the intended SR optimizer rather than selecting a different method silently.
- `run.py` supports SLURM but has no Condor backend, creates OpenMP probes outside the repository, builds over the tracked convert binary, and deletes binaries on local exit. Its proxy copy uses a Purdue path.
- Conversion may silently skip remote-open, tree, or schema failures and still mark a batch successful. Missing theory inputs are padded with unity. Optional input buffers may retain previous-file contents when branches are absent. Confirm the appropriate failure policy and input availability.
- Conversion defaults to updating raw_entries, while comments describe upstream-filtered inputs with externally supplied normalization. Parallel sample merges share the same sample.json.tmp path. Confirm the normalization definition before changing it.
- Physics event weights and theory sums currently use weight_pu without genWeight. Confirm NLO signed-weight handling and original normalization inputs before changing physics weighting.
- PDF member/alpha_s/scale layouts are assumed globally. Inspect source metadata before declaring the nuisance coverage complete.
- Uproot >=5.7 writes dictionaries as RNTuples; combine reads the metadata as TTrees. A specific compatibility bug fix is to write metadata with mktree. Official documentation: https://uproot.readthedocs.io/en/stable/basic.html#writing-ttrees-to-a-file.
- Weight/probability validation is disabled in SR and plotting call sites, and the QCD weight check is commented out while reporting success. Re-enabling previously removed checks needs user confirmation.
- The saved JMS/JMR fit is stale (11.56 fb^-1) and must be regenerated. The fit skips missing samples. Its documentation says GenPart is unavailable while CR branch configuration requires GenPart; inspect the actual schema before selecting a fitting approach.
- Current default model is BDT for five trees. It attempts CUDA and retries CPU on any XGBoostError; the custom decorrelation objective uses NumPy. CPU thread selection may exceed Condor allocations. NN configurations are absent from the current default config.
- Standalone b-veto uses hardcoded old UParT-v2 datasets and a different selection. Confirm ancillary-study scope and physics definitions before running it as part of this campaign.

## Current execution state

Visible question cards work after the user's settings edit and restart. The first input decisions and second physics decisions have been received. Continue source audit, metadata investigation, and software preparation. Production weighting changes remain pending an evidence-based proposal and user confirmation; ordinary campaign execution is already authorized.

## Second decision round: answers received

- MC weighting: investigate Runs metadata, generator weights, and upstream filtering, then propose a coherent full-pipeline correction for confirmation.
- AK4 cleaning: the user explicitly selects cleaning against only the two retained AK8 jets. Apply to nominal and every jet variation, with fresh downstream results.
- SR optimization: the user asks which is reasonable and requests whichever finds the optimum. Compare both on identical input and constraints. Neither current candidate-generation heuristic establishes an unrestricted global optimum. Choose the final objective explicitly before comparing and preserve completion/bound evidence.

## Prepared external calibration inputs and DY evidence

The three explicitly authorized original C-I pileup ROOT files have been obtained under `tmp/lxplus_full_20261001/inputs/pileup/`; recorded hashes are in the campaign log. Their ROOT file signatures have been checked; histogram validation remains for batch workers.

DAS parent metadata resolves the duplicate DY source to the official RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v4 parent. The official McM request matches that parent, but its generator request registers cross_section=1 and filter_efficiency=0.07, without a measured cross section. This does not validate 56.84 or 175.3. Inspect generator run products and source normalization before choosing the alias/value.
