for i in de fr ru zh
do
    python bitext.py --src_lang $i --dataset bucc --seed 42 --cuda --model_checkpoint intfloat/multilingual-e5-base --prompt "query: "
done