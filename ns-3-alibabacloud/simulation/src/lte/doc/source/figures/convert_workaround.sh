#!/bin/bash


mkdir TempOutput
for f in *.dia
do
  echo "converting - $f"
  s=${f%.dia}
  dia --export $s.eps $f
  mv $s.eps TempOutput
done

cd TempOutput
mkdir PNG
for f in *.eps
do
  echo "converting - $f"
  s=${f%.eps}
  convert $f $s.png
  mv $s.png PNG
done
