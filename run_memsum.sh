#!/bin/bash
echo "script to run memsum tasks"

if ls lsf.* 1> /dev/null 2>&1
then
    echo "lsf files do exist"
    echo "removing older lsf files"
    rm lsf.*
fi

if [[ -d runs/ ]]; then echo "removing runs"; rm -r runs/; fi
if [[ -d snnclassifiermetrics/ ]]; then echo "removing snnclassfiermetrics"; rm -r snnclassifiermetrics/; fi
if [[ -d linearclassifiermetrics/ ]]; then echo "removing linearclassifiermetrics"; rm -r linearclassifiermetrics/; fi

module load gcc/8.2.0 python_gpu/3.8.5 eth_proxy
source ../venv_memsum/bin/activate

args=(
    -G ls_lawecon
    -n 4 
    -W 4:00
    -R "rusage[mem=6400]"
)

echo "getting into paraphrase directory"
echo "removing .pyc files"
# find . -name \*.pyc -delete

if [ -z "$1" ]; then echo "CPU mode selected"; fi
while [ ! -z "$1" ]; do
    case "$1" in
	gpu)
	    echo "GPU mode selected"
	    args+=(-R "rusage[ngpus_excl_p=4]")
	    ;;
	intr)
	    echo "Interactive mode selected"
	    args+=(-Is)
	    ;;
    esac
    shift
done

path=../paraphrase/test_corpora/archive

count=0

for eachfile in "$path"/*.txt
do
   echo $eachfile
   ((count++))
   echo $count
   if [ "$count" -gt 0 ]
   then
        bsub "${args[@]}" -oo memsum_$count.out python memsum_extractor.py $eachfile
        # break
   fi
   if [ "$count" -eq 2000 ]
   then
       break
   fi
done


# bsub "${args[@]}" -oo memsum.out python memsum_extractor.py 

