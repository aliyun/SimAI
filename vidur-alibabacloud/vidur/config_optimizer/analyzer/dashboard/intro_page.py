import streamlit as st

from vidur.config_optimizer.analyzer.dashboard.utils import add_small_divider


def render_intro_page():
    st.markdown("## Vidur Demo: Config Explorer")
    add_small_divider()
    st.markdown(
        "Vidur helps you explore how different system configuration parameters affect LLM inference performance. To know more about Vidur refer to our [MLSys24 paper](https://mlsys.org/virtual/2024/poster/2667) and check out the [source code](https://github.com/microsoft/vidur)."
    )
    add_small_divider()
    st.markdown("### Tools")
    st.markdown(
        "We provide four tools to help you analyze the results of Vidur simulation runs:"
    )
    st.markdown("#### Best Config Selection")
    st.markdown(
        "This tool helps you find and visualize variations in best configuration for a given model model under different workloads."
    )
    st.markdown("#### Config Comparison")
    st.markdown(
        'This tool enabled one-on-one performance comparison between different configurations. You can either manually add configuration on this page or use the **"Add to Configuration"** button on other pages to quickly populate the comparison.'
    )
    st.markdown("#### Pareto Curve Analysis")
    st.markdown(
        "This tool helps you visualize the trade-offs between different performance metrics and helps you identify the best configurations."
    )
    st.markdown("#### Cost Analysis")
    st.markdown(
        "This tool helps you select the best configuration based on your SLOs and cost budget. It also provides additional visualizations which allow to see the performance as a function of different config options."
    )
    add_small_divider()
    st.markdown("### Sample Models and Workloads")
    st.markdown(
        "We provide a set of sample models and workloads to help you get started with Vidur. These models and workloads are based on real-world datasets and are designed to help you understand how different configurations affect performance."
    )
    st.markdown("#### Models")
    st.markdown("We provide the following models for you to explore in Vidur:")
    st.markdown(
        """
        | Model Name | Number of Parameters | Number of Layers | Embedding Size | Number of Attention Heads | Attention Type |
        |------------|-------------|-----------------|---------------|--------------------------|---------------|
        | phi-2 | 2.7B | 32| 2560 | 32 | Multi-Head Attention |
        | Llama-2 7B | 7B | 32 | 4096 | 32 | Multi-Head Attention |
        | InternLM 20B | 20B | 60 | 5120 | 40 | Multi-Head Attention |
        | CodeLlama 34B | 34B | 48 | 8192 | 64 | Group-Head Attention |
        | Llama-2 70B | 70B | 80 | 8192 | 64 | Group-Head Attention |
        | Qwen-72B | 72B | 80 | 8192 | 64 | Multi-Head Attention |
        """
    )
    st.markdown("#### Workloads")
    st.markdown(
        "We provide the following three different workloads for you to explore in Vidur:"
    )
    st.markdown(
        """
        | Dataset                       | Content                                           | Num queries | Num prefill tokens (mean, median, P90) | Num decode tokens (mean, median, P90) | PD Ratio (median, std dev) |
        |-------------------------------|---------------------------------------------------|-----------|--------------------------------------|-------------------------------------|----------------------------|
        | LMSys-Chat-1M-4K      | Natural language conversations | 2M        | 686, 417, 1678                       | 197, 139, 484                       | 2.3, 228                   |
        | Arxiv-Summarization-4K | Summarization of arXiv papers  | 28k       | 2588, 2730, 3702                    | 291, 167, 372                       | 15.7, 16                   |
        | Bilingual-Web-Book-4K  | Document-level English–Chinese translation dataset | 33k       | 1067, 1037, 1453                    | 1612, 1601, 2149                    | 0.65, 0.37                 |
        """
    )
    st.markdown("")
    add_small_divider()
    st.markdown("### Citation")
    st.markdown("If you use Vidur in your research, please cite the following paper:")
    st.markdown(
        """
        ```
        @article{agrawal2024vidur,
          title={Vidur: A Large-Scale Simulation Framework For LLM Inference},
          author={Agrawal, Amey and Kedia, Nitin and Mohan, Jayashree and Panwar, Ashish  and Kwatra, Nipun and Gulavani, Bhargav S and Ramjee, Ramachandran and Tumanov, Alexey},
          journal={Proceedings of The Seventh Annual Conference on Machine Learning and Systems, 2024, Santa Clara},
          year={2024}
        }
        ```
        """
    )
