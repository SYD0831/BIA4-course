#!/usr/bin/env bash

SRC="/Users/shenyu/Desktop/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis-master"
DST="/Users/shenyu/git_repo/2025-26-Group-01/classification/data/MPIDB"

mkdir -p "$DST"/{train,val,test}/{falciparum,vivax,ovale}

split_copy () {
  local SP="$1"                               # Falciparum / Vivax / Ovale
  local CLS="$(echo "$SP" | tr 'A-Z' 'a-z')"  # falciparum / vivax / ovale
  local LIST="/tmp/list_${CLS}.txt"

  find "$SRC/$SP/img" -type f \
       \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
       > "$LIST"

  local N="$(wc -l < "$LIST" | tr -d '[:space:]')"
  if [ "$N" -eq 0 ]; then
    echo "WARN: $SP 的 img/ 下未找到图片"
    return
  fi

  awk 'BEGIN{srand(42)} {printf "%f\t%s\n", rand(), $0}' "$LIST" \
    | sort -k1,1n | cut -f2- > "${LIST}.shuf"

  N=$(wc -l < "${LIST}.shuf" | tr -d '[:space:]')
  Ntrain=$(( N * 70 / 100 ))
  Nval=$(( N * 15 / 100 ))
  Ntest=$(( N - Ntrain - Nval ))

  head -n "$Ntrain" "${LIST}.shuf" \
    | xargs -I{} cp -n "{}" "$DST/train/$CLS/"
  sed -n "$((Ntrain+1)),$((Ntrain+Nval))p" "${LIST}.shuf" \
    | xargs -I{} cp -n "{}" "$DST/val/$CLS/"
  sed -n "$((Ntrain+Nval+1)),\$p" "${LIST}.shuf" \
    | xargs -I{} cp -n "{}" "$DST/test/$CLS/"

  echo "$CLS -> train:$Ntrain, val:$Nval, test:$Ntest (total:$N)"
}

split_copy Falciparum
split_copy Vivax
split_copy Ovale

echo
for SPLIT in train val test; do
  printf "[%s]\n" "$SPLIT"
  for C in falciparum vivax ovale; do
    printf "  %-10s: " "$C"
    find "$DST/$SPLIT/$C" -type f | wc -l
  done
done

command -v tree >/dev/null && tree -L 2 "$DST" || true