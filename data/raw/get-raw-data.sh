set -e
set -u
all_lang_pairs=(en-de de-en en-cs cs-en en-ru ru-en en-zh zh-en de-fr fr-de en-ja ja-en uk-en en-uk uk-cs cs-uk en-hr en-ha ha-en en-is is-en)

for lp in ${all_lang_pairs[@]}
do
    src=${lp%%-*}
    tgt=${lp##*-}

    if [[ $lp =~ "is" || $lp =~ "ha" ]]
    then
        sacrebleu -t wmt21 -l $lp --echo src > wmt21.$lp.$src
        sacrebleu -t wmt21 -l $lp --echo ref > wmt21.$lp.$tgt
    else
        sacrebleu -t wmt22 -l $lp --echo src > wmt22.$lp.$src
        sacrebleu -t wmt22 -l $lp --echo ref > wmt22.$lp.$tgt
    fi
done