# CMSJMECalculators

This packages provides an efficient
[ROOT::RDataFrame](https://root.cern/doc/master/classROOT_1_1RDataFrame.html)-friendly
implementation of the recipes for jet and MET variations for the CMS experiment,
for use with samples in the NanoAOD format.
The code was adopted from the [bamboo](https://gitlab.cern.ch/cp3-cms/bamboo)
analysis framework.

**The latest stable release is tag 0.4.0**

**This is a preview to gather feedback (please open an issue with yours),
without any guarantees of stability (including in naming) for now**

## Installation

For using these helpers from python, the recommended solution is to install
the package (in a
[virtual](https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments)
or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
environment) with
```python
pip install git+https://gitlab.cern.ch/cp3-cms/CMSJMECalculators.git
```
[scikit-build](https://scikit-build.readthedocs.io/en/latest/) is used to
compile the C++ components against the available ROOT distribution.

Inside a CMSSW environment, the [`install_cmssw.sh`](install_cmssw.sh) script
can be used:
```bash
wget -q 'https://gitlab.cern.ch/cp3-cms/CMSJMECalculators/-/raw/main/install_cmssw.sh'
source ./install_cmssw.sh
```
if a specific version is needed, the `$VERSION` variable can be set, e.g.
```bash
VERSION=0.1.0 source ./install_cmssw.sh
```

From C++ the package can be installed directly with [CMake](https://cmake.org/),
using the standard commands (after cloning the repository):
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<your-prefix> [other-options] <source-clone>
make
make install
```
This will also install the python modules in
`<your-prefix>/lib/pythonX.Y/site-packages/CMSJMECalculators/`.

## Usage

When installed as a python package or directly with CMake,
the necessary components can be loaded with:
```python
from CMSJMECalculators import loadJMESystematicsCalculators
loadJMESystematicsCalculators()
```
Note that this will load the shared library and headers or dictionary in
[cling](https://root.cern/cling/), the [ROOT](https://root.cern/) interpreter,
so they can from then on also be used in JITted code, e.g. from
[RDataFrame](https://root.cern/doc/master/classROOT_1_1RDataFrame.html).

The variations are calculated by the C++ classes ``JetVariationsCalculator`` and
``FatJetVariationsCalculator`` for the AK4 and AK8 jet JER and JES variations, and
``Type1METVariationsCalculator`` for the Type-1 MET variations, using the standard procedure 
(Type-1 smeared or standard MET is a configuration option).
To use these, an instance should be created (with the C++ interpreter, to make it
available from JITted code), and additional configuration passed by calling
setter methods, e.g. in PyROOT:
```python
from CMSJMECalculators import loadJMESystematicsCalculators
loadJMESystematicsCalculators()
import ROOT as gbl

from CMSJMECalculators import config as calcConfigs
configCls = calcConfigs.JetVariations # or METVariations, FatJetVariations
jsonFile = "jet_jerc.json.gz" # path to your json file 
config = configCls(jsonFile)
config.jetAlgo = "AK4PFchs"
config.jecTag = "Summer19UL18_V5_MC"
config.jecLevel = "L1L2L3Res" 
# Optional parameters
# config.jesUncertainties = ["Total"]
# config.jerTag = "Summer19UL18_JRV2_MC"
# config.splitJER = False
# config.jsonFileSmearingTool = "jer_smear.json.gz" #path to your json file with jer smearing tool
# config.smearingToolName = "JERSmear"
# etc.

calc = gbl.JetVariationsCalculator(config.create())
# or:
# calc = gbl.Type1METVariationsCalculator(config.create())
# calc = gbl.FatJetVariationsCalculator(config.create())
```
The varied jet pt's and masses can be obtained by calling the ``produce`` method
with the per-event quantities, converted to
[`ROOT::VecOps::RVec`](https://root.cern/doc/master/classROOT_1_1VecOps_1_1RVec.html):
```python
from CMSJMECalculators.utils import toRVecFloat, toRVecInt
jetVars = calc.produce(toRVecFloat(tree.Jet_pt), toRVecFloat(tree.Jet_eta), ...)
```
Since the full list of arguments can be long, and depends on a few parameters
(for data the MC branches are not there, and not needed, and MET needs a few
additional inputs), a helper function is provided, which can be used as follows:
```python
from CMSJMECalculators.utils import getJetMETArgs
jetVars = calc.produce(*getJetMETArgs(tree, isMC=True, forMET=False))
```
This will return an object that contains all the variations, e.g.
`jetVars.pt(0)` will return the `RVec` with new nominal jet PTs.
The corresponding names of the variations, which depend on the configuration,
can be retrieved from the calculator by calling its `available()` method.

### From (JITted) RDataFrame

When constructing the RDataFrame graph from python, the calculator needs to be
constructed directly from the cling interpreter, such that it is available in
the global C++ namespace for JITted code:
```python
gbl.gROOT.ProcessLine("JetVariationsCalculator myJetVarCalc = 
JetVariationsCalculator::create(jsonFile, jetType, jecTag, jecLevel...);")
calc = getattr(gbl, "myJetVarCalc")
```
the second line retrieves a reference from PyROOT, such that the configuration
methods can be called as above.

Inside the RDataFrame graph the varied jet pt's and masses can be defined as
a new column:
```python
df.Define("ak4JetVars", "myJetVarCalc.produce(Jet_pt, Jet_eta, Jet_phi, ...)")
```
(the full set of arguments is not reproduced here, but can be found from the
`utils.getJetMETargs` method; since RDataFrame uses `RVec` internally
no conversion is needed).

### From C++

The PyROOT example above relies on the automatically generated bindings, so
the C++ equivalent is almost identical, and straigthforward to obtain.
When calling the `produce` method outside RDataFrame, most of the arguments
may need to be converted to `RVec`, which fortunately supports all common
kinds of array interfaces.

```python
JetVariationsCalculator myJetVarCalc = JetVariationsCalculator::create(jsonFile,
jetType, jecTag, jecLevel...);
auto df = df.Define("ak4JetVars",[&](const RVec<float> &Jet_pt, ...) ->
JetVariationsCalculator::result_t { return myJetVarCalc.produce(Jet_pt, ...);
},{"Jet_pt", ...});
```

### Json files

The JEC and JER json files can be downloaded from the corresponding repositories at
```/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/```.
The json file with the jer smearing tool is available at
```/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/jer_smear.json.gz```.

## Testing and development

A set of [pytest](https://docs.pytest.org/en/6.2.x/)-based tests are included,
to make sure the implementation stays consistent with the POG-provided python
version in [nanoAOD-tools](https://github.com/cms-nanoAOD/nanoAOD-tools).
The tests compare the contents of the pt and mass branches for all variations.
They can be run with
```python
pytest tests
```
or, inside a CMSSW environment where python2 is the default
```python
python3 -m pytest tests
```

TODO make tests python2-compatible(?), expand, scripts for larger tests samples?
