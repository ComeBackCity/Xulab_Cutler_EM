taus=(0.05 0.10 0.15 0.20 0.35 0.5 0.65 0.8 0.85 0.88 0.9 0.95)
for tau in "${taus[@]}"
do
  mkdir "out/$tau"
done
files=($( ls imgs/out/test_dir ))
for file in "${files[@]}"
do
  for tau in "${taus[@]}"
  do
    echo "Running inference"
    CUDA_VISIBLE_DEVICES=5,6 python demo.py --img-path "imgs/out/test_dir/$file" \
      --N 3 --tau "$tau" --vit-arch base --patch-size 8 \
      --output_path "out/$tau/$file"
    echo "Inference done"
  done
done