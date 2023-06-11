This is data format we convert for all dataset

```
<data_name>
\ Images
    \ train
        \ img1.jpg
        \ img2.jpg
        ....
    \ val
        ...
    \ test
        ...
\ gt
    \ train
        \ img1.txt
        \ img2.txt
        ...
    \ val
        ...
    \ test
        ...
```

Each `txt` file will has following annotaiton:

```
cls xn yn wn hn x1 y1 x2 y2 ... xt yt text
```