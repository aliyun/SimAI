# 最新ニュース
### SimCCLのアップデート
[2025/06] SimCCLのコードが最初に[SimCCL](https://github.com/aliyun/SimAI/tree/SimCCL)ブランチで公開され、まもなくSimCCLリポジトリでリリースされます。
<div align="center">
🎯 <b>イベントとコミュニティ活動</b> 🎯

### 📅 今後のイベント

| 日付 | イベント | 場所 | 内容 | 形式 |
|:----:|:------|:---------|:--------|:----:|
| 未定 | SimAI 2.0 | 🌐 オンライン | SimAI 2.0のリリース | 💻 バーチャル |

### 🌟 過去のイベント

| 日付 | イベント | 場所 | 内容 | 形式 |
|:----:|:------|:---------|:--------|:----:|
| 2025年6月4日 | SimAIコミュニティ第1回ワークショップ | 📍 北京大学 | コミュニティ貢献者による3つの講演 | 🎓 現地 |
| 2025年5月24日 | 第28回Chinasysワークショップ | 📍 重慶大学 | SimAIに関する招待講演 | 🎓 現地 |
| 2024年12月27日 | SimAI技術発表会 | 📍 北京航空航天大学 | SimAI技術共有とディスカッション | 🎓 現地 |
| 2024年12月6日 | HKUST技術ワークショップ | 📍 香港科技大学(広州) | SimAI技術共有とディスカッション | 🎓 現地 |
| 2024年12月5日 | [Bench'24カンファレンス](https://mp.weixin.qq.com/s/STic_E12xMhZRxhzK9wRnw) | 📍 広州 | SimAIチュートリアルと詳細セッション | 🎓 現地 |
| 2024年11月26日 | SimAIコミュニティライブストリーム | 🌐 オンライン | インタラクティブな技術ディスカッションとデモ（400人以上参加） | 💻 バーチャル |
| 2024年11月15日 | 技術ワークショップ | 📍 千島湖 | SimAIオフライン技術交流会 | 🎯 現地 |
| 2024年10月18日 | ゲスト講義 | 📍 復旦大学 | SimAIチュートリアルと公開講座 | 🎓 現地 |
| 2024年9月24-26日 | CCF HPC China 2024 | 📍 武漢 | SimAI紹介と技術発表 | 🎤 カンファレンス |
</div>

---

# 目次
- [SimAI 概要](#simai-概要)
  - [はじめに](#はじめに)
  - [コンポーネント](#コンポーネント)
  - [シナリオ](#シナリオ)
  - [引用](#引用)
- [使い方](#使い方)
  - [セットアップ](#セットアップ)
    - [ソースコードから](#ソースコードから)
  - [SimAI-Analyticalの使い方](#simai-analyticalの使い方)
  - [SimAI-Simulationの使い方](#simai-simulationの使い方)

# SimAI 概要
## はじめに

**SimAI**は、業界初のフルスタック・高精度な大規模AIトレーニング用**Sim**ulator（**シミュレーター**）です。フレームワーク、集合通信、ネットワーク層など、LLMトレーニングプロセス全体を詳細にモデリング・シミュレーションします。この包括的なアプローチにより、エンドツーエンドのパフォーマンスデータが提供され、研究者は以下のことが可能になります：

- トレーニングプロセスの詳細分析
- 特定条件下でのAIタスクの時間消費の評価
- 以下を含む様々なアルゴリズム最適化によるE2Eパフォーマンスゲインの評価：
  - フレームワークのパラメータ設定
  - 集合通信アルゴリズム
  - NCCL環境変数
  - ネットワーク転送プロトコル
  - 輻輳制御アルゴリズム
  - 適応ルーティングアルゴリズム
  - スケールアップ/アウトネットワークトポロジの変更
  - ...

## コンポーネント

<pre>
        |--- <a href="https://github.com/aliyun/aicb">AICB</a>
SimAI --|--- <a href="https://github.com/aliyun/SimCCL">SimCCL</a>
        |--- <a href="https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud">astra-sim-alibabacloud</a>
        |--- <a href="https://github.com/aliyun/ns-3-alibabacloud">ns-3-alibabacloud</a>
</pre>

純粋なシミュレーション能力を基盤に、SimAIは4つのコンポーネント（[aicb](https://github.com/aliyun/aicb)、[SimCCL](https://github.com/aliyun/SimCCL)、[astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)、[ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud)）からなる多機能なフルスタックツールキットに進化しました。これらのコンポーネントは、様々な方法で組み合わせて異なる機能を実現できます。以下に、SimAIの6つの主な使用シナリオを示します。この強力なツールでさらに多くの可能性を探求することをお勧めします。

以下はSimAIシミュレータのアーキテクチャ図です：
![SimAI_Arc](./docs/images/SimAI_Arc.png)

astra-sim-alibabacloudは[astra-sim](https://github.com/astra-sim/astra-sim/tree/ASTRA-sim-1.0)を拡張したものです。astra-simチームの素晴らしい仕事とオープンソースへの貢献に感謝します。私たちはNCCLアルゴリズムを統合し、いくつかの新機能を追加しました。

## シナリオ

SimAIは、さまざまなシミュレーション要件を満たすために、3つの主要な動作モードをサポートしています：

**SimAI-Analytical**は、バス帯域幅（busbw）を使用して集合通信時間を見積もることにより、ネットワーク通信の詳細を抽象化し、高速なシミュレーションを提供します。現在、ユーザー定義のbusbwをサポートしていますが、自動busbw計算機能はまもなく登場予定です。

**SimAI-Simulation**は、きめ細かいネットワーク通信モデリングを備えたフルスタックシミュレーションを提供します。NS3や他のネットワークシミュレータ（現在はNS3がオープンソース化されています）を活用して、すべての通信動作を詳細にシミュレーションし、実際のトレーニング環境を高忠実に再現することを目指しています。

**SimAI-Physical**（ベータ版）は、CPU RDMAクラスタ環境向けの物理トラフィック生成を可能にします。このモードはNCCLのようなトラフィックパターンを生成し、LLMトレーニング中のNICの動作を詳細に研究することができます。現在、内部テスト段階です。

| シナリオ | 説明 | コンポーネントの組み合わせ |
|----------|-------------|------------------------|
| 1. AICBテストスイート | AICBテストスイートを使用してGPUクラスタで通信パターンを実行 | [AICB](https://github.com/aliyun/aicb) |
| 2. AICB/AIOBワークロード | トレーニングプロセスの計算/通信パターンをモデル化してワークロードを生成 | [AICB](https://github.com/aliyun/aicb) |
| 3. 集合通信分析 | 集合通信操作をポイントツーポイント通信セットに分解 | [SimCCL](https://github.com/aliyun/SimCCL) |
| 4. GPUなしでの集合通信 | 非GPUクラスタでRDMA集合通信トラフィックを実行 | [AICB](https://github.com/aliyun/aicb) + [SimCCL](https://github.com/aliyun/SimCCL) + [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)(physical) |
| 5. SimAI-Analytical | 任意のサーバーで迅速なAICBワークロード分析とシミュレーションを実施（基盤となるネットワークの詳細は無視） | [AICB](https://github.com/aliyun/aicb) + [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)(analytical) |
| 6. SimAI-Simulation | 任意のサーバーで完全なシミュレーションを実行 | [AICB](https://github.com/aliyun/aicb) + [SimCCL](https://github.com/aliyun/SimCCL) + [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)(simulation) + [ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud) |


## 引用

SimAIの研究はNSDI'25 Springに採択されました。詳細については、以下の論文をご参照ください：

*SimAI: Unifying Architecture Design and Performance Tuning for Large-Scale Large Language Model Training with Scalability and Precision.*

[[pdf](https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf)] / [[slides](./docs/SimAI_Intro_Online.pdf)] / [[video](https://n.dingtalk.com/dingding/live-room/index.html?roomId=OF5BkBUXVxmgsK7x&liveUuid=305736cd-aa70-498b-8003-2b471a53decd)]

SimAIを基盤とした革新的な研究や拡張を奨励します。ディスカッションのために私たちのコミュニティグループに参加するか、メールでお問い合わせください。技術的なサポートを提供する場合があります。

# クイックスタート

以下に簡単な例を示します。SimAIの完全なチュートリアルはこちらにあります：[**SimAI@Tutorial**](./docs/Tutorial.md)、[**aicb@Tutorial**](https://github.com/aliyun/aicb/blob/master/training/tutorial.md)、[SimCCL@Tutorial]、[ns-3-alibabacloud@Tutorial]

## セットアップ

以下の手順に従って、環境を迅速にセットアップし、SimAIを実行できます。

### ソースコードから

以下のコードは、Ubuntu 20.04のGCC/G++ 9.4.0、python 3.8.10で正常にテストされています。

公式のUbuntu 20.04イメージを使用し、ninjaをインストールしないでください。

（ワークロードの生成には、NGCコンテナイメージを直接利用することをお勧めします。）

```bash
# リポジトリをクローン
$ git clone https://github.com/aliyun/SimAI.git
$ cd ./SimAI/

# サブモジュールをクローン
$ git submodule update --init --recursive
# 最新のコミットを使用することを確認
$ git submodule update --remote

# SimAI-Analyticalをコンパイル
$ ./scripts/build.sh -c analytical

# SimAI-Simulation (ns3)をコンパイル
$ ./scripts/build.sh -c ns3

```

## SimAI-Analyticalの使い方

```bash
$  ./bin/SimAI_analytical -w example/workload_analytical.txt -g 9216 -g_p_s 8 -r test- -busbw example/busbw.yaml
```

バス帯域幅を自動で計算するには、次のコマンドを試してください：
```bash
$  ./bin/SimAI_analytical -w ./example/workload_analytical.txt -g 9216  -nv 360 -nic 48.5 -n_p_s 8 -g_p_s 8 -r example-
```

## SimAI-Simulationの使い方

```bash
# ネットワークトポロジを作成
$ python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps

# 実行
$ AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator -t 16 -w ./example/microAllReduce.txt -n ./Spectrum-X_128g_8gps_100Gbps_A100 -c astra-sim-alibabacloud/inputs/config/SimAI.conf

```

# お問い合わせ

ご不明な点がございましたら、Gang Lu (yunding.lg@alibaba-inc.com) または Qingxu Li (qingxu.lqx@alibaba-inc.com) までメールでお問い合わせください。

SimAIコミュニティのチャットグループへの参加を歓迎します。左がDingTalkグループ、右がWeChatグループです。

<div style="display: flex; justify-content: flex-start; align-items: center; gap: 20px; margin-left: 20px;">
    <img src="./docs/images/simai_dingtalk.jpg" alt="SimAI DingTalk" style="width: 300px; height: auto;">
    <img src="./docs/images/simai_wechat.jpg" alt="SimAI WeChat" style="width: 300px; height: auto;">
</div>

<br/>
