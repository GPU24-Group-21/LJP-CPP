prog="./ljp"
infile="Rap_2_LJP.in"

# find any .in file and set it as the input file
if [ -f *.in ]; then
    infile=$(ls *.in)
fi

clear

# Run the program with the given arguments
echo "Running $prog with input file $infile\n"

# run cpu version
echo "Running CPU version"
$prog $infile 0 0

# # run gpu version
# echo "\n ------------------------------------ "
# echo "Running GPU version"
# $prog $infile 0 1