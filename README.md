# tehai-extractor
雀魂のスクリーンショットから手牌をextractする

# How to Use
1. `./data`に牌の画像データを入れておく
2. 以下を実行
```shell
docker build -t tehai .
docker run --rm tehai image/target.png
```
