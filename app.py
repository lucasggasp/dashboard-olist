"""
Dashboard Olist (E-commerce brasileiro) - FESA, Eletiva II, Trabalho N2.

Como rodar:
    pip install -r requirements.txt
    streamlit run app.py

Fonte dos dados (escolha na barra lateral):
  - "Pasta local": pasta com os CSVs do Kaggle (padrão: ./data)
  - "Kaggle (kagglehub)": baixa automaticamente (precisa de internet)
Mapas: coloque o arquivo br_states.json na mesma pasta do app (ou envie pela barra lateral).
Sem o GeoJSON os mapas viram gráficos de barras.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Olist | Dashboard", page_icon="📦", layout="wide")

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
AZUL, VERDE, LARANJA, VERMELHO = "#4C72B0", "#55A868", "#DD8452", "#C44E52"
STATUS_EXCLUIDOS_VENDAS = ["canceled", "unavailable"]
ORDEM_CATEGORIAS = ["No prazo", "Atraso de até 7 dias", "Atraso superior a 7 dias"]
DIAS_SEMANA = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"}

ARQUIVOS = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "product_category_name_translation.csv",
]
COLUNAS_DATA_ORDERS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
STOPWORDS_BASICAS = (
    "de a o que e do da em um para é com uma os no se na por mais as dos como mas foi ao ele das "
    "tem à seu sua ou ser quando muito há já está eu também só pelo pela até isso ela entre era "
    "depois mesmo ter seus quem nas me esse eles estão você tinha foram essa num meu às minha "
    "têm numa pelos elas havia seja qual será nós tenho lhe deles essas esses pelas este fosse "
    "dele tu te vocês vos lhes meus minhas teu tua teus tuas nosso nossa nossos nossas dela delas "
    "esta estes estas aquele aquela aqueles aquelas isto aquilo estou está estamos estão estive "
    "esteve estivemos estiveram estava estávamos estavam"
).split()


def brl(x):
    """Formata número como moeda brasileira (R$ 1.234,56)."""
    return "R$ " + f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def inteiro(x):
    return f"{int(x):,}".replace(",", ".")


def pct(x, casas=1):
    return f"{x:.{casas}f}%".replace(".", ",")


# ----------------------------------------------------------------------------
# Carga e preparação dos dados (mesma lógica do notebook)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Baixando o dataset do Kaggle...")
def baixar_kaggle():
    import kagglehub

    return kagglehub.dataset_download("olistbr/brazilian-ecommerce")


@st.cache_data(show_spinner="Carregando e preparando os dados...")
def preparar(pasta: str):
    def ler(nome, **kw):
        return pd.read_csv(os.path.join(pasta, nome), **kw)

    orders = ler("olist_orders_dataset.csv", parse_dates=COLUNAS_DATA_ORDERS)
    items = ler("olist_order_items_dataset.csv")
    customers = ler("olist_customers_dataset.csv")
    reviews = ler(
        "olist_order_reviews_dataset.csv",
        parse_dates=["review_creation_date", "review_answer_timestamp"],
    )
    products = ler("olist_products_dataset.csv")
    trad = ler("product_category_name_translation.csv")

    # Uma avaliação por pedido (a mais recente)
    reviews_unique = (
        reviews.sort_values("review_answer_timestamp")
        .drop_duplicates("order_id", keep="last")
        .reset_index(drop=True)
    )

    # Data de compra + estado do cliente por pedido
    pedido_info = orders[["order_id", "customer_id", "order_purchase_timestamp"]].merge(
        customers[["customer_id", "customer_state"]], on="customer_id", how="left"
    )
    reviews_unique = reviews_unique.merge(
        pedido_info[["order_id", "order_purchase_timestamp", "customer_state"]],
        on="order_id",
        how="left",
    )

    # Receita, frete e nº de itens por pedido
    valor_pedido = items.groupby("order_id", as_index=False).agg(
        receita=("price", "sum"),
        frete=("freight_value", "sum"),
        n_itens=("order_item_id", "count"),
    )
    valor_pedido["frete_pct"] = valor_pedido["frete"] / valor_pedido["receita"]

    # Base de pedidos entregues
    entregues = orders[
        (orders["order_status"] == "delivered") & orders["order_delivered_customer_date"].notna()
    ].copy()
    entregues["tempo_entrega_dias"] = (
        entregues["order_delivered_customer_date"] - entregues["order_purchase_timestamp"]
    ).dt.days
    dif = (
        entregues["order_delivered_customer_date"].dt.normalize()
        - entregues["order_estimated_delivery_date"].dt.normalize()
    ).dt.days
    entregues["dias_atraso"] = dif.clip(lower=0)
    entregues["atrasado"] = (dif > 0).astype(int)
    entregues["categoria_entrega"] = pd.cut(
        dif, bins=[-np.inf, 0, 7, np.inf], labels=ORDEM_CATEGORIAS
    )
    entregues = (
        entregues.merge(
            customers[["customer_id", "customer_unique_id", "customer_state"]],
            on="customer_id",
            how="left",
        )
        .merge(reviews_unique[["order_id", "review_score"]], on="order_id", how="left")
        .merge(valor_pedido, on="order_id", how="left")
    )

    # Base de itens de pedidos entregues
    itens_entregues = items.merge(
        entregues[
            [
                "order_id",
                "order_purchase_timestamp",
                "customer_state",
                "tempo_entrega_dias",
                "dias_atraso",
                "atrasado",
                "categoria_entrega",
                "review_score",
            ]
        ],
        on="order_id",
        how="inner",
    )

    # Bases de vendas (tudo que não foi cancelado/indisponível)
    pedidos_vendas = (
        pedido_info[
            pedido_info["order_id"].isin(
                orders.loc[~orders["order_status"].isin(STATUS_EXCLUIDOS_VENDAS), "order_id"]
            )
        ]
        .merge(valor_pedido, on="order_id", how="inner")
        .drop(columns="customer_id")
    )
    itens_vendas = (
        items[["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"]]
        .merge(
            pedidos_vendas[["order_id", "order_purchase_timestamp", "customer_state"]],
            on="order_id",
            how="inner",
        )
        .merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
        .merge(trad, on="product_category_name", how="left")
    )
    itens_vendas["categoria"] = (
        itens_vendas["product_category_name_english"]
        .fillna(itens_vendas["product_category_name"])
        .fillna("sem_categoria")
    )

    # Q8: nota por categoria (cada pedido conta uma vez por categoria)
    q8_base = (
        items[["order_id", "product_id"]]
        .drop_duplicates()
        .merge(products[["product_id", "product_category_name"]], on="product_id")
        .merge(trad, on="product_category_name")
        .drop_duplicates(["order_id", "product_category_name_english"])
        .merge(
            reviews_unique[["order_id", "review_score", "order_purchase_timestamp", "customer_state"]],
            on="order_id",
        )
    )

    # Q9: retenção (histórico completo de cada cliente)
    q9_pedidos = (
        orders[~orders["order_status"].isin(STATUS_EXCLUIDOS_VENDAS)]
        .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id")
        .merge(reviews_unique[["order_id", "review_score"]], on="order_id", how="left")
        .sort_values("order_purchase_timestamp")
    )
    q9_clientes = pd.DataFrame({"pedidos": q9_pedidos.groupby("customer_unique_id")["order_id"].nunique()})
    q9_clientes["retido"] = q9_clientes["pedidos"] > 1
    q9_clientes["nota_1o"] = q9_pedidos.drop_duplicates("customer_unique_id").set_index(
        "customer_unique_id"
    )["review_score"]
    q9_clientes = q9_clientes.dropna(subset=["nota_1o"])
    q9_clientes["nota_1o"] = q9_clientes["nota_1o"].astype(int)

    # Contagem de pedidos por mês (completude do dataset, independe dos filtros)
    def por_mes(df):
        s = df.groupby(df["order_purchase_timestamp"].dt.to_period("M")).size()
        s.index = s.index.astype(str)
        return s

    return {
        "entregues": entregues,
        "itens_entregues": itens_entregues,
        "pedidos_vendas": pedidos_vendas,
        "itens_vendas": itens_vendas,
        "reviews_unique": reviews_unique,
        "q8_base": q8_base,
        "q9_clientes": q9_clientes,
        "vendas_por_mes": por_mes(pedidos_vendas),
        "entregues_por_mes": por_mes(entregues),
        "data_min": orders["order_purchase_timestamp"].min().date(),
        "data_max": orders["order_purchase_timestamp"].max().date(),
        "estados": sorted(customers["customer_state"].dropna().unique().tolist()),
    }


def carregar_geojson(upload, caminho):
    """Lê o GeoJSON dos estados; retorna None se não estiver disponível ou for inválido."""
    try:
        if upload is not None:
            gj = json.load(upload)
        elif caminho and Path(caminho).exists():
            gj = json.loads(Path(caminho).read_text(encoding="utf-8"))
        else:
            return None
        for f in gj["features"]:
            f["properties"]["PK_sigla"] = str(f["properties"]["PK_sigla"]).upper()
        return gj
    except Exception as erro:
        st.sidebar.warning(f"GeoJSON inválido ({erro}). Usando barras no lugar dos mapas.")
        return None


# ----------------------------------------------------------------------------
# Barra lateral: fonte dos dados e filtros
# ----------------------------------------------------------------------------
st.sidebar.title("📦 Olist")
st.sidebar.caption("FESA · Eletiva II · Trabalho N2")

fonte = st.sidebar.radio(
    "Fonte dos dados",
    ["Pasta local", "Kaggle (kagglehub)"],
    index=0 if Path("data").exists() else 1,
)
if fonte == "Pasta local":
    pasta = st.sidebar.text_input("Pasta com os CSVs", value="data")
else:
    try:
        pasta = baixar_kaggle()
    except Exception as erro:
        st.error(f"Não foi possível baixar do Kaggle: {erro}")
        st.stop()

faltando = [a for a in ARQUIVOS if not (Path(pasta) / a).exists()]
if faltando:
    st.error(f"Arquivos não encontrados em `{pasta}`: {', '.join(faltando)}")
    st.info(
        "Baixe o dataset em https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce, "
        "extraia os CSVs em uma pasta e informe o caminho na barra lateral."
    )
    st.stop()

P = preparar(str(pasta))

with st.sidebar.expander("🗺️ Mapa dos estados (GeoJSON)"):
    up = st.file_uploader("br_states.json", type=["json", "geojson"])
    caminho_geo = st.text_input("ou caminho do arquivo", value="br_states.json")
GEOJSON = carregar_geojson(up, caminho_geo)

st.sidebar.markdown("### Filtros")
periodo = st.sidebar.date_input(
    "Período da compra",
    value=(P["data_min"], P["data_max"]),
    min_value=P["data_min"],
    max_value=P["data_max"],
)
if not isinstance(periodo, (tuple, list)) or len(periodo) != 2:
    st.info("Selecione a data inicial e a final no filtro de período.")
    st.stop()
estados_sel = st.sidebar.multiselect("Estados (destino do cliente)", P["estados"], default=P["estados"])
if not estados_sel:
    st.warning("Selecione ao menos um estado.")
    st.stop()
MIN_PEDIDOS_MES = st.sidebar.number_input(
    "Mín. de pedidos para um mês ser considerado completo (Q2 e Q11)", 0, 5000, 300, step=50
)

secao = st.sidebar.radio(
    "Seção",
    [
        "Visão geral",
        "Moda, média e mediana",
        "Vendas e receita (Q1–Q3)",
        "Entregas e logística (Q4–Q6)",
        "Satisfação e avaliações (Q7–Q9)",
        "Estados, sazonalidade e vendedores (Q10–Q12)",
    ],
)


def filtrar(df, col_data="order_purchase_timestamp"):
    ini = pd.Timestamp(periodo[0])
    fim = pd.Timestamp(periodo[1]) + pd.Timedelta(days=1)
    m = (df[col_data] >= ini) & (df[col_data] < fim)
    if len(estados_sel) < len(P["estados"]):
        m &= df["customer_state"].isin(estados_sel)
    return df[m]


ENT = filtrar(P["entregues"])
ITE = filtrar(P["itens_entregues"])
PV = filtrar(P["pedidos_vendas"])
IV = filtrar(P["itens_vendas"])
REV = filtrar(P["reviews_unique"])
Q8B = filtrar(P["q8_base"])

if PV.empty or ENT.empty:
    st.warning("Nenhum pedido encontrado com os filtros atuais.")
    st.stop()


# ----------------------------------------------------------------------------
# Helpers de gráficos
# ----------------------------------------------------------------------------
def barras_h(df, x, y, titulo, cor=AZUL, rotulo_x=None):
    fig = px.bar(df, x=x, y=y, orientation="h", title=titulo, labels={x: rotulo_x or x, y: ""})
    fig.update_traces(marker_color=cor)
    fig.update_yaxes(autorange="reversed")  # 1º do ranking no topo
    return fig


def mapa_ou_barras(df_estado, coluna, titulo, escala, rotulo):
    if GEOJSON is not None:
        fig = px.choropleth(
            df_estado,
            geojson=GEOJSON,
            locations="customer_state",
            featureidkey="properties.PK_sigla",
            color=coluna,
            color_continuous_scale=escala,
            labels={coluna: rotulo},
            title=titulo,
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    else:
        d = df_estado.sort_values(coluna, ascending=False)
        fig = barras_h(d, coluna, "customer_state", titulo, rotulo_x=rotulo)
    st.plotly_chart(fig)


# ----------------------------------------------------------------------------
# Cabeçalho
# ----------------------------------------------------------------------------
st.title("📦 Olist · Dashboard de E-commerce")
st.caption(
    f"Período: {periodo[0]:%d/%m/%Y} a {periodo[1]:%d/%m/%Y} · "
    f"{len(estados_sel)} de {len(P['estados'])} estados · "
    "Receita = soma de `price` dos itens (sem frete) · Pedido de venda = status diferente de "
    "cancelado/indisponível · Pedido entregue = status `delivered` com data de entrega."
)


# ----------------------------------------------------------------------------
# Seções
# ----------------------------------------------------------------------------
def secao_visao_geral():
    c = st.columns(4)
    c[0].metric("Receita", brl(PV["receita"].sum()))
    c[1].metric("Pedidos de venda", inteiro(len(PV)))
    c[2].metric("Ticket médio", brl(PV["receita"].mean()))
    c[3].metric("Itens vendidos", inteiro(len(IV)))
    c = st.columns(4)
    c[0].metric("Pedidos entregues", inteiro(len(ENT)))
    c[1].metric("Tempo médio de entrega", f"{ENT['tempo_entrega_dias'].mean():.1f} dias".replace(".", ","))
    c[2].metric("Entregas atrasadas", pct(ENT["atrasado"].mean() * 100))
    c[3].metric("Nota média", f"{ENT['review_score'].mean():.2f}".replace(".", ","))

    mensal = (
        PV.assign(ano_mes=PV["order_purchase_timestamp"].dt.to_period("M").astype(str))
        .groupby("ano_mes")
        .agg(receita=("receita", "sum"), pedidos=("order_id", "count"))
        .reset_index()
    )
    fig = px.area(mensal, x="ano_mes", y="receita", title="Receita mensal", labels={"ano_mes": "", "receita": "Receita (R$)"})
    fig.update_traces(line_color=AZUL)
    st.plotly_chart(fig)

    col1, col2 = st.columns(2)
    por_estado = (
        PV.groupby("customer_state").agg(receita=("receita", "sum")).reset_index()
    )
    with col1:
        mapa_ou_barras(por_estado, "receita", "Receita por estado", "Blues", "Receita (R$)")
    with col2:
        d = ENT["categoria_entrega"].value_counts().reindex(ORDEM_CATEGORIAS).reset_index()
        d.columns = ["categoria", "pedidos"]
        fig = px.pie(
            d, names="categoria", values="pedidos", hole=0.45, title="Pontualidade das entregas",
            color="categoria",
            color_discrete_map=dict(zip(ORDEM_CATEGORIAS, [VERDE, LARANJA, VERMELHO])),
        )
        st.plotly_chart(fig)


def secao_central():
    st.markdown(
        "Medidas de tendência central sobre os **pedidos entregues**. "
        "Nos histogramas o eixo é cortado no percentil 99 apenas para leitura; as estatísticas usam todos os dados."
    )
    opcoes = {
        "Tempo de entrega (dias)": ("tempo_entrega_dias", "Dias entre compra e entrega", False),
        "Frete por pedido (R$)": ("frete", "Frete (R$)", False),
        "Valor dos itens por pedido (R$)": ("receita", "Valor (R$)", False),
        "Nota da avaliação": ("review_score", "Nota", True),
    }
    nome = st.radio("Variável", list(opcoes), horizontal=True)
    col, rotulo, discreto = opcoes[nome]
    s = ENT[col].dropna()
    moda = s.mode().iloc[0]
    m = st.columns(3)
    m[0].metric("Moda", f"{moda:.2f}")
    m[1].metric("Média", f"{s.mean():.2f}")
    m[2].metric("Mediana", f"{s.median():.2f}")

    if discreto:
        vc = s.astype(int).value_counts().sort_index()
        fig = px.bar(x=vc.index.astype(str), y=vc.values, labels={"x": rotulo, "y": "Pedidos"})
        fig.update_traces(marker_color=AZUL)
    else:
        s_plot = s[s <= s.quantile(0.99)]
        fig = px.histogram(x=s_plot, nbins=40, labels={"x": rotulo})
        fig.update_traces(marker_color=AZUL, opacity=0.85)
        ymax = np.histogram(s_plot, bins=40)[0].max()
        for valor, nm, cor, dash in [
            (s.mean(), "Média", "red", "dash"),
            (s.median(), "Mediana", "green", "solid"),
            (moda, "Moda", "orange", "dot"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=[valor, valor], y=[0, ymax], mode="lines",
                    name=f"{nm} ({valor:.2f})", line=dict(color=cor, dash=dash, width=3),
                )
            )
        fig.update_layout(yaxis_title="Pedidos")
    st.plotly_chart(fig)


def secao_vendas():
    q1, q2, q3 = st.tabs(["Q1 · Categorias", "Q2 · Receita mensal", "Q3 · Valor dos pedidos"])

    with q1:
        st.subheader("Categorias com maior volume de vendas e contribuição na receita")
        n = st.slider("Quantidade de categorias no ranking", 5, 20, 10, key="q1_n")
        cat = (
            IV.groupby("categoria")
            .agg(itens_vendidos=("order_item_id", "count"), pedidos=("order_id", "nunique"), receita=("price", "sum"))
            .assign(
                pct_receita=lambda d: d["receita"] / d["receita"].sum() * 100,
                pct_volume=lambda d: d["itens_vendidos"] / d["itens_vendidos"].sum() * 100,
            )
            .sort_values("itens_vendidos", ascending=False)
            .reset_index()
        )
        top = cat.head(n).copy()
        top.index = range(1, len(top) + 1)
        em_comum = len(set(cat.nlargest(n, "receita")["categoria"]) & set(top["categoria"]))
        st.info(
            f"As {n} categorias mais vendidas concentram **{pct(top['pct_receita'].sum())}** da receita e "
            f"**{pct(top['pct_volume'].sum())}** dos itens. {em_comum} delas também estão no top {n} por receita."
        )
        c1, c2 = st.columns(2)
        c1.plotly_chart(barras_h(top, "itens_vendidos", "categoria", "Itens vendidos", AZUL, "Itens"))
        c2.plotly_chart(barras_h(top, "pct_receita", "categoria", "% da receita total", VERDE, "% da receita"))
        st.dataframe(top.round(2))

    with q2:
        st.subheader("Evolução da receita mensal")
        mensal = (
            PV.assign(ano_mes=PV["order_purchase_timestamp"].dt.to_period("M"))
            .groupby("ano_mes")
            .agg(pedidos=("order_id", "count"), receita=("receita", "sum"))
        )
        mensal = mensal.reindex(pd.period_range(mensal.index.min(), mensal.index.max(), freq="M"), fill_value=0)
        mensal.index = mensal.index.astype(str)
        mensal.index.name = "ano_mes"
        mensal["completo"] = [P["vendas_por_mes"].get(m, 0) >= MIN_PEDIDOS_MES for m in mensal.index]
        var = (mensal["receita"] / mensal["receita"].shift(1) - 1) * 100
        mensal["variacao_pct"] = var.where(mensal["completo"] & mensal["completo"].shift(fill_value=False))
        comp = mensal[mensal["completo"]]

        if comp.empty:
            st.warning("Nenhum mês completo no período selecionado.")
        else:
            c = st.columns(3)
            c[0].metric("Receita no período", brl(mensal["receita"].sum()))
            c[1].metric("Melhor mês", comp["receita"].idxmax(), brl(comp["receita"].max()), delta_color="off")
            c[2].metric("Pior mês", comp["receita"].idxmin(), brl(comp["receita"].min()), delta_color="off")
        graf = mensal.reset_index()
        graf["situação"] = np.where(graf["completo"], "Completo", "Incompleto")
        fig = px.bar(
            graf, x="ano_mes", y="receita", color="situação",
            color_discrete_map={"Completo": AZUL, "Incompleto": "lightgray"},
            labels={"ano_mes": "", "receita": "Receita (R$)"},
            title="Receita mensal (cinza = mês incompleto, fora do ranking)",
        )
        st.plotly_chart(fig)
        if not comp.empty:
            a, b = st.columns(2)
            a.markdown("**3 melhores meses**")
            a.dataframe(comp.nlargest(3, "receita")[["pedidos", "receita", "variacao_pct"]].round(2))
            b.markdown("**3 piores meses**")
            b.dataframe(comp.nsmallest(3, "receita")[["pedidos", "receita", "variacao_pct"]].round(2))
        with st.expander("Série mensal completa"):
            st.dataframe(mensal.round(2))

    with q3:
        st.subheader("Distribuição do valor dos pedidos")
        v = PV["receita"]
        c = st.columns(4)
        c[0].metric("Mínimo", brl(v.min()))
        c[1].metric("Máximo", brl(v.max()))
        c[2].metric("Média", brl(v.mean()))
        c[3].metric("Mediana", brl(v.median()))
        st.caption(
            f"Média / mediana = {v.mean() / v.median():.2f} "
            f"({'distribuição assimétrica à direita' if v.mean() > v.median() else 'sem assimetria à direita'})."
        )

        txt = st.text_input("Limiares de ordem de magnitude (R$), separados por vírgula", "10, 100, 1000")
        try:
            limiares = sorted({float(x.replace(",", ".")) for x in txt.replace(";", " ").split() if x} or {10, 100, 1000})
        except ValueError:
            limiares = [10.0, 100.0, 1000.0]
            st.warning("Valores inválidos; usando 10, 100 e 1000.")
        lim = pd.DataFrame(
            {
                "limiar (R$)": limiares,
                "abaixo": [int((v < l).sum()) for l in limiares],
                "igual ou acima": [int((v >= l).sum()) for l in limiares],
            }
        )
        lim["% abaixo"] = lim["abaixo"] / len(v) * 100
        lim["% igual ou acima"] = lim["igual ou acima"] / len(v) * 100

        faixas = ["Menos de R$ 10", "R$ 10 a 99,99", "R$ 100 a 999,99", "R$ 1.000 a 9.999,99", "R$ 10.000 ou mais"]
        fx = (
            pd.cut(v, bins=[0, 10, 100, 1000, 10000, np.inf], labels=faixas, right=False)
            .value_counts().reindex(faixas).to_frame("pedidos")
        )
        fx["percentual"] = fx["pedidos"] / fx["pedidos"].sum() * 100

        c1, c2 = st.columns(2)
        with c1:
            bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 50)
            cont, bordas = np.histogram(v, bins=bins)
            centros = np.sqrt(bordas[:-1] * bordas[1:])
            fig = go.Figure(go.Bar(x=centros, y=cont, marker_color=AZUL, name="Pedidos"))
            fig.update_xaxes(type="log", title="Valor do pedido (R$, escala log)")
            fig.update_yaxes(title="Pedidos")
            for l in limiares:
                fig.add_vline(x=l, line_dash="dash", line_color="red", opacity=0.6)
            fig.add_trace(go.Scatter(x=[v.median()] * 2, y=[0, cont.max()], mode="lines", name=f"Mediana ({brl(v.median())})", line=dict(color="green", width=3)))
            fig.add_trace(go.Scatter(x=[v.mean()] * 2, y=[0, cont.max()], mode="lines", name=f"Média ({brl(v.mean())})", line=dict(color="orange", width=3)))
            fig.update_layout(title="Distribuição do valor dos pedidos (escala log)")
            st.plotly_chart(fig)
        with c2:
            fig = px.bar(fx.reset_index(names="faixa"), x="faixa", y="percentual", title="% de pedidos por faixa de valor", labels={"faixa": "", "percentual": "% dos pedidos"})
            fig.update_traces(marker_color=LARANJA)
            st.plotly_chart(fig)
        a, b = st.columns(2)
        a.dataframe(lim.round(2), hide_index=True)
        b.dataframe(fx.round(2))


def secao_entregas():
    q4, q5, q6 = st.tabs(["Q4 · Tempo por estado", "Q5 · Frete × tempo", "Q6 · Pontualidade"])

    with q4:
        st.subheader("Tempo médio de entrega por estado e entregas atrasadas")
        est = (
            ENT.groupby("customer_state")
            .agg(pedidos=("order_id", "count"), tempo_medio_dias=("tempo_entrega_dias", "mean"), pedidos_atrasados=("atrasado", "sum"))
            .assign(pct_atrasados=lambda d: d["pedidos_atrasados"] / d["pedidos"] * 100)
            .reset_index()
            .sort_values("tempo_medio_dias", ascending=False)
        )
        c = st.columns(3)
        c[0].metric("Pedidos entregues", inteiro(len(ENT)))
        c[1].metric("Atrasados", inteiro(ENT["atrasado"].sum()), pct(ENT["atrasado"].mean() * 100), delta_color="off")
        c[2].metric("Tempo médio (Brasil)", f"{ENT['tempo_entrega_dias'].mean():.1f} dias".replace(".", ","))
        a, b = st.columns(2)
        with a:
            mapa_ou_barras(est, "tempo_medio_dias", "Tempo médio de entrega por estado", "Viridis_r", "Dias")
        with b:
            mapa_ou_barras(est, "pedidos_atrasados", "Entregas atrasadas por estado", "Reds", "Pedidos atrasados")
        st.dataframe(est.round(2), hide_index=True)

    with q5:
        st.subheader("Correlação entre custo de frete e tempo de entrega")
        corr = ITE["freight_value"].corr(ITE["tempo_entrega_dias"])
        intensidade = "fraca" if abs(corr) < 0.3 else "moderada" if abs(corr) < 0.7 else "forte"
        st.metric("Correlação de Pearson (nível do item)", f"{corr:.2f}".replace(".", ","), intensidade, delta_color="off")
        amostra = ITE.sample(n=min(10000, len(ITE)), random_state=42)
        fig = px.scatter(amostra, x="freight_value", y="tempo_entrega_dias", opacity=0.3,
                         labels={"freight_value": "Custo do frete (R$)", "tempo_entrega_dias": "Tempo de entrega (dias)"},
                         title="Frete × tempo de entrega (amostra de até 10.000 itens)")
        if len(amostra) > 1:
            a1, b1 = np.polyfit(amostra["freight_value"], amostra["tempo_entrega_dias"], 1)
            xs = np.array([amostra["freight_value"].min(), amostra["freight_value"].max()])
            fig.add_trace(go.Scatter(x=xs, y=a1 * xs + b1, mode="lines", name="Tendência", line=dict(color="red", width=3)))
        st.plotly_chart(fig)

        frete_est = (
            ITE.groupby("customer_state")["freight_value"].mean().reset_index(name="frete_medio").sort_values("frete_medio", ascending=False)
        )
        a, b = st.columns(2)
        with a:
            st.plotly_chart(barras_h(frete_est.head(10), "frete_medio", "customer_state", "10 estados com maior frete médio", VERMELHO, "Frete médio (R$)"))
        with b:
            mapa_ou_barras(frete_est, "frete_medio", "Custo médio de frete por estado", "Reds", "Frete médio (R$)")

    with q6:
        st.subheader("Entregas no prazo × atrasadas")
        dist = ENT["categoria_entrega"].value_counts().reindex(ORDEM_CATEGORIAS).to_frame("pedidos")
        dist["percentual"] = dist["pedidos"] / dist["pedidos"].sum() * 100
        c = st.columns(3)
        for col, (cat, linha) in zip(c, dist.iterrows()):
            col.metric(cat, pct(linha["percentual"], 2), f"{inteiro(linha['pedidos'])} pedidos", delta_color="off")
        fig = px.bar(dist.reset_index(names="categoria"), x="categoria", y="percentual", color="categoria",
                     color_discrete_map=dict(zip(ORDEM_CATEGORIAS, [VERDE, LARANJA, VERMELHO])),
                     labels={"categoria": "", "percentual": "% dos pedidos entregues"},
                     title="Distribuição das entregas em relação ao prazo estimado")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)


@st.cache_data(show_spinner="Calculando termos frequentes...")
def termos_frequentes(comentarios: pd.Series, n: int = 15):
    try:
        import nltk
        from nltk.corpus import stopwords

        nltk.download("stopwords", quiet=True)
        base = stopwords.words("portuguese")
    except Exception:
        base = STOPWORDS_BASICAS
    sw = [w for w in base if w not in ("não", "nem", "sem")] + ["produto", "pedido", "comprei"]
    try:
        vet = CountVectorizer(ngram_range=(1, 2), stop_words=sw, min_df=20)
        mat = vet.fit_transform(comentarios)
    except ValueError:
        return pd.Series(dtype=int)
    return pd.Series(np.asarray(mat.sum(axis=0)).ravel(), index=vet.get_feature_names_out()).nlargest(n)


def secao_satisfacao():
    q7, q8, q9 = st.tabs(["Q7 · Nota × atraso", "Q8 · Categorias e comentários", "Q9 · Satisfação × retenção"])

    with q7:
        st.subheader("Notas: entregas no prazo × com atraso")
        d = ENT.dropna(subset=["review_score"]).copy()
        d["grupo"] = np.where(d["atrasado"] == 1, "Com atraso", "No prazo")
        if d["grupo"].nunique() < 2:
            st.warning("É preciso haver pedidos no prazo e atrasados para comparar.")
        else:
            media = d.groupby("grupo")["review_score"].mean()
            neg = d.groupby("grupo")["review_score"].apply(lambda s: (s <= 2).mean() * 100)
            p = stats.mannwhitneyu(
                d.loc[d["atrasado"] == 0, "review_score"], d.loc[d["atrasado"] == 1, "review_score"]
            ).pvalue
            c = st.columns(4)
            c[0].metric("Nota média · no prazo", f"{media['No prazo']:.2f}".replace(".", ","))
            c[1].metric("Nota média · atrasado", f"{media['Com atraso']:.2f}".replace(".", ","))
            c[2].metric("Notas 1-2 · no prazo", pct(neg["No prazo"]))
            c[3].metric("Notas 1-2 · atrasado", pct(neg["Com atraso"]))
            st.caption(
                f"Diferença de {media['No prazo'] - media['Com atraso']:.2f} pontos · teste de Mann-Whitney p = {p:.1e}"
                + (" (estatisticamente significativa)." if p < 0.05 else " (não significativa a 5%).")
            )
            ct = pd.crosstab(d["grupo"], d["review_score"].astype(int), normalize="index") * 100
            long = ct.reset_index().melt(id_vars="grupo", var_name="Nota", value_name="pct")
            fig = px.bar(long, x="Nota", y="pct", color="grupo", barmode="group",
                         color_discrete_map={"No prazo": VERDE, "Com atraso": VERMELHO},
                         labels={"pct": "% dos pedidos", "grupo": ""},
                         title="Distribuição das notas: no prazo vs. com atraso")
            st.plotly_chart(fig)

    with q8:
        st.subheader("Piores categorias e padrão nos comentários negativos")
        minimo = st.slider("Mínimo de avaliações por categoria", 10, 500, 100, step=10)
        cats = Q8B.groupby("product_category_name_english")["review_score"].agg(["count", "mean"])
        cats = cats[cats["count"] >= minimo].sort_values("mean")
        a, b = st.columns(2)
        with a:
            if cats.empty:
                st.warning("Nenhuma categoria com avaliações suficientes.")
            else:
                pior = cats.head(10).reset_index()
                st.plotly_chart(barras_h(pior, "mean", "product_category_name_english", "10 piores categorias (nota média)", VERMELHO, "Nota média"))
                st.caption(f"Média geral: {Q8B['review_score'].mean():.2f}".replace(".", ","))
        with b:
            neg = REV.loc[REV["review_score"] <= 2, "review_comment_message"].dropna()
            termos = termos_frequentes(neg)
            if termos.empty:
                st.warning("Poucos comentários negativos para extrair termos.")
            else:
                t = termos.reset_index()
                t.columns = ["termo", "frequencia"]
                st.plotly_chart(barras_h(t, "frequencia", "termo", "Termos mais frequentes (notas 1-2)", LARANJA, "Frequência"))
        if not cats.empty:
            with st.expander("Tabela completa de categorias"):
                st.dataframe(cats.rename(columns={"count": "avaliações", "mean": "nota média"}).round(2))

    with q9:
        st.subheader("Relação entre satisfação e retenção de clientes")
        st.caption("Cliente = `customer_unique_id`; retido = 2+ pedidos. Usa o histórico completo (não depende dos filtros).")
        cl = P["q9_clientes"]
        recompra = (cl.groupby("nota_1o")["retido"].mean() * 100).reindex(range(1, 6))
        p = stats.chi2_contingency(pd.crosstab(cl["nota_1o"], cl["retido"]))[1]
        nota_ret = cl.groupby("retido")["nota_1o"].mean()
        c = st.columns(4)
        c[0].metric("Clientes retidos", pct(cl["retido"].mean() * 100, 2))
        c[1].metric("Nota 1º pedido · retidos", f"{nota_ret.get(True, float('nan')):.2f}".replace(".", ","))
        c[2].metric("Nota 1º pedido · compra única", f"{nota_ret.get(False, float('nan')):.2f}".replace(".", ","))
        c[3].metric("Recompra: nota 1-2 → 4-5", f"{recompra.loc[[1, 2]].mean():.1f}% → {recompra.loc[[4, 5]].mean():.1f}%".replace(".", ","))
        st.caption(f"Associação testada por qui-quadrado: p = {p:.1e}.")
        fig = px.bar(recompra.reset_index(), x="nota_1o", y="retido",
                     labels={"nota_1o": "Nota do primeiro pedido", "retido": "% que voltou a comprar"},
                     title="Recompra por nota do primeiro pedido")
        fig.update_traces(marker_color=AZUL)
        st.plotly_chart(fig)


def secao_estados_sazon_vend():
    q10, q11, q12 = st.tabs(["Q10 · Estados problemáticos", "Q11 · Sazonalidade", "Q12 · Vendedores"])

    with q10:
        st.subheader("Estados com maiores problemas e impacto de uma intervenção")
        est = (
            ENT.groupby("customer_state")
            .agg(pedidos=("order_id", "count"), taxa_atraso=("atrasado", "mean"), nota_media=("review_score", "mean"),
                 frete_pct_mediano=("frete_pct", "median"), frete_medio=("frete", "mean"))
            .reset_index()
        )
        if len(est) < 3:
            st.info("Selecione ao menos 3 estados para calcular o score de problema.")
        else:
            for col, sinal in [("taxa_atraso", 1), ("nota_media", -1), ("frete_pct_mediano", 1)]:
                est[f"z_{col}"] = sinal * (est[col] - est[col].mean()) / est[col].std()
            est["score_problema"] = est[["z_taxa_atraso", "z_nota_media", "z_frete_pct_mediano"]].mean(axis=1)
            est = est.sort_values("score_problema", ascending=False)
            st.caption("Score = média dos z-scores de taxa de atraso, nota (invertida) e frete/valor dos itens (mediana).")
            st.dataframe(est[["customer_state", "pedidos", "taxa_atraso", "nota_media", "frete_pct_mediano", "score_problema"]].head(10).round(3), hide_index=True)

            cols = st.columns(3)
            for c, (col, titulo) in zip(cols, [("taxa_atraso", "Taxa de atraso"), ("nota_media", "Nota média"), ("frete_pct_mediano", "Frete / valor dos itens")]):
                d = est.sort_values(col, ascending=(col == "nota_media"))
                c.plotly_chart(barras_h(d, col, "customer_state", titulo, AZUL, titulo), key=f"q10_{col}")

            st.markdown("#### Simulação de intervenção")
            st.caption("Nos 5 piores estados (mín. 500 pedidos) a taxa de atraso cai até a média dos demais. "
                       "Correlação, não causalidade: leia como ordem de grandeza.")
            nota_atr = ENT.loc[ENT["atrasado"] == 1, "review_score"].mean()
            nota_ok = ENT.loc[ENT["atrasado"] == 0, "review_score"].mean()
            ganho = nota_ok - nota_atr
            piores = est[est["pedidos"] >= 500].head(5)["customer_state"].tolist()
            meta = ENT.loc[~ENT["customer_state"].isin(piores), "atrasado"].mean()
            if not piores or np.isnan(meta) or np.isnan(ganho):
                st.info("Não há estados suficientes (com 500+ pedidos) para simular com os filtros atuais.")
            else:
                sim = est[est["customer_state"].isin(piores)].copy()
                sim["pedidos_evitados_atraso"] = ((sim["taxa_atraso"] - meta).clip(lower=0) * sim["pedidos"]).round(0)
                sim["nova_nota"] = sim["nota_media"] + sim["pedidos_evitados_atraso"] * ganho / sim["pedidos"]
                c = st.columns(3)
                c[0].metric("Meta de atraso", pct(meta * 100))
                c[1].metric("Pedidos atrasados evitados", inteiro(sim["pedidos_evitados_atraso"].sum()))
                c[2].metric("Ganho de nota por pedido evitado", f"{ganho:.2f}".replace(".", ","))
                st.dataframe(sim[["customer_state", "pedidos", "taxa_atraso", "pedidos_evitados_atraso", "nota_media", "nova_nota"]].round(3), hide_index=True)

    with q11:
        st.subheader("Sazonalidade de vendas, atrasos e satisfação")
        d = ENT.assign(
            ano_mes=ENT["order_purchase_timestamp"].dt.to_period("M").astype(str),
            mes=ENT["order_purchase_timestamp"].dt.month,
            dia_semana=ENT["order_purchase_timestamp"].dt.dayofweek,
        )
        mensal = (
            d.groupby("ano_mes")
            .agg(pedidos=("order_id", "count"), receita=("receita", "sum"), taxa_atraso=("atrasado", "mean"), nota_media=("review_score", "mean"))
            .reset_index()
        )
        validos = [m for m in mensal["ano_mes"] if P["entregues_por_mes"].get(m, 0) >= MIN_PEDIDOS_MES]
        mensal = mensal[mensal["ano_mes"].isin(validos)]
        if mensal.empty:
            st.warning("Nenhum mês completo no período selecionado.")
        else:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Pedidos por mês", "Taxa de atraso", "Nota média"), vertical_spacing=0.08)
            fig.add_trace(go.Scatter(x=mensal["ano_mes"], y=mensal["pedidos"], mode="lines+markers", line_color=AZUL), row=1, col=1)
            fig.add_trace(go.Scatter(x=mensal["ano_mes"], y=mensal["taxa_atraso"], mode="lines+markers", line_color=VERMELHO), row=2, col=1)
            fig.add_trace(go.Scatter(x=mensal["ano_mes"], y=mensal["nota_media"], mode="lines+markers", line_color=VERDE), row=3, col=1)
            fig.update_layout(height=650, showlegend=False)
            st.plotly_chart(fig)

            dv = d[d["ano_mes"].isin(validos)]
            metrica = st.radio("Métrica", ["pedidos", "taxa_atraso", "nota_media"], horizontal=True, key="q11_metrica")
            sazonal = dv.groupby("mes").agg(pedidos=("order_id", "count"), taxa_atraso=("atrasado", "mean"), nota_media=("review_score", "mean"))
            semana = dv.groupby("dia_semana").agg(pedidos=("order_id", "count"), taxa_atraso=("atrasado", "mean"), nota_media=("review_score", "mean"))
            semana.index = semana.index.map(DIAS_SEMANA)
            a, b = st.columns(2)
            fa = px.bar(sazonal.reset_index(), x="mes", y=metrica, title="Por mês do ano", labels={"mes": "Mês"})
            fa.update_traces(marker_color=AZUL)
            a.plotly_chart(fa)
            fb = px.bar(semana.reset_index(names="dia"), x="dia", y=metrica, title="Por dia da semana", labels={"dia": ""})
            fb.update_traces(marker_color=LARANJA)
            b.plotly_chart(fb)
            st.caption("Os anos não têm a mesma cobertura (alguns meses aparecem em mais de um ano): "
                       "para o padrão anual compare principalmente as taxas; para volume, use a série mensal.")

            limite = mensal["taxa_atraso"].mean() + mensal["taxa_atraso"].std()
            crit = mensal[mensal["taxa_atraso"] > limite]
            st.markdown(f"**Meses críticos** (taxa de atraso acima de {pct(limite * 100)} = média + 1 desvio padrão)")
            if crit.empty:
                st.write("Nenhum mês acima do limite.")
            else:
                st.dataframe(crit[["ano_mes", "pedidos", "taxa_atraso", "nota_media"]].round(3), hide_index=True)

    with q12:
        st.subheader("Concentração de receita e qualidade dos vendedores")
        vend = ITE.groupby("seller_id").agg(receita=("price", "sum"), pedidos=("order_id", "nunique"))
        qual = (
            ITE.drop_duplicates(["seller_id", "order_id"])
            .groupby("seller_id")
            .agg(nota_media=("review_score", "mean"), taxa_atraso=("atrasado", "mean"))
        )
        vend = vend.join(qual).reset_index().sort_values("receita", ascending=False).reset_index(drop=True)
        vend["receita_acum_pct"] = vend["receita"].cumsum() / vend["receita"].sum()
        total = len(vend)
        n50 = int((vend["receita_acum_pct"] <= 0.5).sum() + 1)
        n80 = int((vend["receita_acum_pct"] <= 0.8).sum() + 1)
        c = st.columns(3)
        c[0].metric("Vendedores", inteiro(total))
        c[1].metric("Geram 50% da receita", inteiro(n50), pct(n50 / total * 100, 2), delta_color="off")
        c[2].metric("Geram 80% da receita", inteiro(n80), pct(n80 / total * 100, 2), delta_color="off")

        a, b = st.columns(2)
        with a:
            fig = px.line(x=np.arange(1, total + 1) / total * 100, y=vend["receita_acum_pct"] * 100,
                          labels={"x": "% de vendedores", "y": "% da receita acumulada"},
                          title="Curva de Pareto: receita por vendedor")
            fig.add_hline(y=80, line_dash="dash", line_color="gray")
            st.plotly_chart(fig)
        with b:
            top = vend.head(n80)[["nota_media", "taxa_atraso"]].describe().round(3)
            demais = vend.iloc[n80:][["nota_media", "taxa_atraso"]].describe().round(3)
            st.markdown("**Vendedores que concentram 80% da receita**")
            st.dataframe(top)
            st.markdown("**Demais vendedores**")
            st.dataframe(demais)

        st.markdown("#### Vendedores de risco")
        k1, k2, k3 = st.columns(3)
        min_ped = k1.number_input("Mín. de pedidos", 1, 1000, 30)
        lim_nota = k2.slider("Nota média abaixo de", 1.0, 5.0, 3.8, step=0.1)
        lim_atraso = k3.slider("Taxa de atraso acima de", 0.0, 1.0, 0.15, step=0.01)
        risco = vend[(vend["pedidos"] >= min_ped) & ((vend["nota_media"] < lim_nota) | (vend["taxa_atraso"] > lim_atraso))]
        st.write(f"**{inteiro(len(risco))}** vendedores relevantes com problema de qualidade "
                 f"({pct(risco['receita'].sum() / vend['receita'].sum() * 100)} da receita).")
        st.dataframe(risco.drop(columns="receita_acum_pct").head(50).round(3), hide_index=True)


{
    "Visão geral": secao_visao_geral,
    "Moda, média e mediana": secao_central,
    "Vendas e receita (Q1–Q3)": secao_vendas,
    "Entregas e logística (Q4–Q6)": secao_entregas,
    "Satisfação e avaliações (Q7–Q9)": secao_satisfacao,
    "Estados, sazonalidade e vendedores (Q10–Q12)": secao_estados_sazon_vend,
}[secao]()
