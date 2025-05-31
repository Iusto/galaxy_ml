
# 🚀 AI プロジェクトポートフォリオ

## 📌 プロジェクト概要

本プロジェクトは、Pythonおよび機械学習技術を活用して、天体銀河の画像を分類（Classification）するAIモデルの開発を目的としています。初期にはSeleniumを用いてGoogle画像検索から銀河の写真を収集し、手動でラベル付けを行って学習に活用しました。しかし、約3000枚という少ない画像数ではモデル性能に限界があり、実際に楕円銀河（E）、渦巻銀河（S）、棒渦巻銀河（SB）に分類した初期モデルの精度は55%にとどまりました。

## 🔍 データ収集およびラベリングプロセス

より良いデータを探す中でGalaxy Zooプロジェクトのデータセットを発見し、活用することに決定しました。ただし、このデータセットも完全にラベル付けされていたわけではなかったため、チームメンバー4人が約4000枚の画像を分担して手動でラベル付けを行い、その後、残りのデータに対して自動ラベル付けシステムを構築しました。結果として以下のような大規模なラベル付きデータセットが完成しました：

* 楕円銀河（E）：59,071枚
* 渦巻銀河（S）：53,334枚
* 棒渦巻銀河（SB）：51,008枚

## ⚗️ データ前処理と学習戦略

* `ImageDataGenerator` によるデータ拡張
* ResNet50を用いた転移学習（Transfer Learning）
* `ReduceLROnPlateau` による学習率の減衰調整
* SGD(momentum=0.9) オプティマイザの使用
* `BatchNormalization` による学習の安定化
* クラス不均衡への対応として `class_weight` を適用

約4000枚の手動ラベル付きデータで学習されたモデルは、**検証データで約75%の精度**を達成しました。特に渦巻銀河と棒渦巻銀河は構造的に非常に類似しているため分類が難しく、この過程で前処理やラベル品質がモデル性能に与える影響の大きさを実感しました。

### 📊 モデル性能比較表

| モデル               | 精度   | Loss | 特記事項                                |
|--------------------|--------|------|----------------------------------------|
| ベースライン CNN       | 55%   | 0.91 | 手動ラベル付け3000枚、基本的な構造               |
| 改善モデル (ResNet50) | 75%   | 0.48 | データ拡張、転移学習、class_weightの適用 |

## 🧠 振り返り

* データが少ない状況での学習の限界を体感し、「良質なデータ」の重要性を理解
* 手動ラベル作業の大変さを体感し、効率的なラベル自動化手法の必要性を感じた
* 転移学習とハイパーパラメータ調整を通して、限られたデータでも一定以上の性能を実現可能であると実証

## 🛠 技術スタック

* 言語：Python
* データ処理：Pandas, NumPy, OpenCV
* モデリング：TensorFlow, Keras, Scikit-learn, PyTorch
* クローリング：Selenium
* 開発環境：PyCharm, Google Colab

## 📎 開発環境と実行

本プロジェクトは主に **Google Colab** 上で実行・実験を行い、クラウドGPUを積極的に活用して並列学習を行いました。ローカル環境ではPyCharmを用いてデバッグやコード整理を行いました。

## 🗂️ フォルダ構成

```bash
project-root/
├── data/                # 元の画像とラベル済みデータ
├── notebooks/           # 実験用Jupyterノートブック
├── models/              # 学習済みモデル保存用
├── src/                 # 学習および推論コード
│   ├── preprocessing/   # 前処理スクリプト
│   ├── training/        # モデル学習モジュール
│   ├── inference/       # 予測ロジック
├── utils/               # ユーティリティ関数
├── README.md
```

## 💻 サンプルコード（CNN）

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

## 🔗 データセットリンク

[Google Drive ダウンロードリンク](https://drive.google.com/file/d/1IJxSUsAFV3cUPBROa9uWaJI6zQm6UgcC/view?usp=sharing)

## 📸 予測結果

![performance_comparison](https://github.com/user-attachments/assets/00d29b8c-d845-4886-af7f-f30d0b84d17e)  
[性能比較グラフ]

![confusion_matrix](https://github.com/user-attachments/assets/74a8f6cf-6f9b-4e7c-8c87-03b67c2b5b48)  
[Confusion Matrix]

```bash
[入力画像]              → [予測結果]
spiral_001.jpg         → 渦巻銀河 (S)
barred_spiral_004.jpg  → 棒渦巻銀河 (SB)
elliptical_100.jpg     → 楕円銀河 (E)
```

## 📅 今後の改善方針

* より精密なラベル自動化アルゴリズムの構築
* 軽量な転移学習モデル（MobileNetなど）の実験
* 学習曲線可視化およびConfusion Matrixを用いた性能分析
* Kaggleなど外部ベンチマークとの比較
