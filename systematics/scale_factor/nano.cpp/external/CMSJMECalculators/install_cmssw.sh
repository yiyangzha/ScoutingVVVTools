# no shebang, must be sourced

# Check if in CMSSW
if [ -z "$CMSSW_BASE" ]; then
  echo "You must use this package inside a CMSSW environment"
  return 1
fi

## deduce source location from the script name
if [[ -z "${ZSH_NAME}" ]]; then
  thisscript="$(readlink -f ${BASH_SOURCE})"
else
  thisscript="$(readlink -f ${0})"
fi
pipinstall="$(dirname ${thisscript})/.python"

local pyvers=""
if [ ! "${PYTHON}" ]; then
  # check if this is a python2 or python3 release
  which python2 > /dev/null 2> /dev/null
  if [ $? -ne 0 ]; then
    PYTHON=python
    pyvers=py2
  else
    python2 -c 'import FWCore.ParameterSet as cms' > /dev/null 2> /dev/null
    if [ $? -ne 0 ]; then 
      PYTHON=python3
      pyvers=py3
    else
      PYTHON=python2
      pyvers=py2
    fi
  fi
else
  pyvers="py$(${PYTHON} -c 'import sys; print(sys.version_info[0])')"
fi

toolname="${pyvers}-cmsjmecalculators"

# Check if it is already installed
scram tool info "${toolname}" > /dev/null 2> /dev/null
if [ $? -eq 0 ]; then
  echo "--> ${toolname} already installed"
  return 0
fi

installpath="${CMSSW_BASE}/install/${toolname}"
if [ -d "${installpath}" ]; then
  echo "--> Install path ${installpath} exists, please remove and try again if you want to reinstall"
  return 1
fi

pymajmin=$(${PYTHON} -c 'import sys; print(".".join(str(num) for num in sys.version_info[:2]))')

echo "--> Installing as ${toolname} with python=${PYTHON} (${pymajmin}) into ${installpath}"

origin="git+https://gitlab.cern.ch/cp3-cms/CMSJMECalculators.git"
if [ "${VERSION}" ]; then
  origin="${origin}@${VERSION}"
fi

# First, download and install pip, if needed
local bk_pythonpath="${PYTHONPATH}"
local bk_path="${PATH}"
local bk_tmpdir="${TMPDIR}"
TMPDIR="${CMSSW_BASE}/tmp"
( ${PYTHON} -m pip --version && ${PYTHON} -m pip download -v "${origin}" ) > /dev/null 2> /dev/null
if [ $? -ne 0 ]; then
  echo "--> No working pip found, bootstrapping in ${pipinstall}"
  [ -d "${pipinstall}" ] || mkdir "${pipinstall}"
  if [ ! -f "${pipinstall}/bin/pip" ]; then
    wget -q -O "${pipinstall}/get-pip.py" "https://bootstrap.pypa.io/pip/${pymajmin}/get-pip.py"
    ${PYTHON} "${pipinstall}/get-pip.py" --prefix="${pipinstall}" --ignore-installed
  fi
  PYTHONPATH="${pipinstall}/lib/python${pymajmin}/site-packages:${PYTHONPATH}"
  PATH="${pipinstall}/bin:${PATH}"
  ${PYTHON} -m pip install --prefix="${pipinstall}" --ignore-installed setuptools_scm scikit-build 'cmake>=3.11'
fi

echo "--> Installing CMSJMECalculators from ${origin}"
mkdir -p ${installpath}
${PYTHON} -m pip install --prefix="${installpath}" --ignore-installed "${origin}"
toolversion=$(${PYTHON} -m pip show "${origin}" | grep Version | sed 's/Version: //')

if [ -d "${pipinstall}" ]; then
  rm -rf "${pipinstall}"
fi
PYTHONPATH="${bk_pythonpath}"
PATH="${bk_path}"
TMPDIR="${bk_tmpdir}"

pyversu=$(echo "${pyvers}" | tr 'a-z' 'A-Z')
pypathname="PYTHONPATH"
if [[ "${pyvers}" == "py3" ]]; then
  pypathname="PYTHON3PATH"
fi
toolfile="${installpath}/${toolname}.xml"
cat <<EOF_TOOLFILE >"${toolfile}"
<tool name="${toolname}" version="${toolversion}">
  <info url="https://gitlab.cern.ch/cp3-cms/CMSJMECalculators"/>
  <client>
    <environment name="${pyversu}_CMSJMECALCULATORS_BASE" default="${installpath}"/>
    <runtime name="LD_LIBRARY_PATH" value="\$${pyversu}_CMSJMECALCULATORS_BASE/lib" type="path"/>
    <runtime name="${pypathname}"   value="\$${pyversu}_CMSJMECALCULATORS_BASE/lib/python${pymajmin}/site-packages" type="path"/>
    <runtime name="PATH"            value="\$${pyversu}_CMSJMECALCULATORS_BASE/bin" type="path"/>
  </client>
</tool>
EOF_TOOLFILE

echo "--> Updating environment"
scram setup "${toolfile}"
cmsenv

echo "--> ${toolname} is installed."
