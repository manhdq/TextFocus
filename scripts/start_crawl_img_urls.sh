
python utils/data_tools/crawl_img_urls.py \
    --out-path datasets/statue_queries_results.csv \
    --checkpoint 0 \
    --queries-path datasets/statue_queries.txt \
    --acceptable-formats jpg,jpeg,png \
    --format jpg \
    --type photo \
    --extract-metadata \
    --language English \
    --chromedriver /usr/bin/chromedriver \
    --related-images \
    --no-download \
    --silent-mode \
    --limit 400 \
