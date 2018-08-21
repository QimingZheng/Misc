#========================================================================
# Setup script for 6.823

export MIT6823_HOME=/mnt/nvmebeegfs/zqm/mit/6.823-lab
#export SVNROOT=file://${MIT6823_HOME}/svnroot/$USER
export LAB0ROOT=${MIT6823_HOME}/lab0handout
export LAB1ROOT=${MIT6823_HOME}/lab1handout
export LAB2ROOT=${MIT6823_HOME}/lab2handout
export LAB3ROOT=${MIT6823_HOME}/lab3handout
export LAB4ROOT=${MIT6823_HOME}/lab4handout

export PIN_HOME=/mnt/nvmebeegfs/zqm/mit/software/pintools
export PIN_ROOT=${PIN_HOME}
export PIN_KIT=${PIN_HOME}
export LIBCONFIGPATH=/usr/local/
export PATH=${PIN_HOME}:$PATH

export MACHTYPE="linux"

if [ "$MACHTYPE" == "linux" ]
then

  echo ""
  echo " -----------------------------------------------------------"
  echo " This is an Athena/Linux machine. "
  echo " Setting up 6.823 Spring 2016 tools. "
  echo " -----------------------------------------------------------"
  echo ""

  echo ""

else 
 
  echo ""
  echo " -----------------------------------------------------------"
  echo " This is an Athena/Solaris machine. Please log in to an "
  echo " Athena/Linux machine"
  echo " -----------------------------------------------------------"
  echo ""

fi
