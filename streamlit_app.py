import streamlit as st
import pandas as pd
import random
import pymarket as pm
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import datetime
from collections import OrderedDict
import pprint

wheelings = [
    [0, 0, 0,  0.87,  0.7,  1.2],
    [0, 0, 0,  0.7,   1.2, 0.87],
    [0, 0, 0,  0.7,  0.87,  1.2],
    [0, 0, 0,    0,     0,    0],
    [0, 0, 0,    0,     0,    0],
    [0, 0, 0,    0,     0,    0],
]

p2p_mech_net_profit = 0
p2p_mech_avg_profit = 0

def p2p_egat_mechanism(bids, p_coef=0.5, r=None) -> (pm.TransactionManager, dict):
    """Computes all the trades using a P2P random trading
    process inspired in [1].

    Parameters
    ----------
    bids: pd.DataFrame
        Collection of bids that will trade.
        Precondition: a user participates only in one
        side of the market, i.e, it cannot sell and buy in
        the same run.
    p_coef: float
        coefficient to calculate the trading price as a convex
        combination of the price of the seller and the price of
        the buyer. If 1, the seller gets all the profit and if 0,
        the buyer gets all the profit.
    r: np.random.RandomState
        Random state to generate stochastic values. If None,
        then the outcome of the market will be different on
        each run.

    Returns
    -------
    trans : TransactionManger
        Collection of all the transactions that ocurred in the market

    extra : dict
        Extra information provided by the mechanisms.
        Keys:

        * trading_list: list of list of tuples of all the pairs that traded in each round.

    Notes
    -------
    [1] Blouin, Max R., and Roberto Serrano. "A decentralized market with
    common values uncertainty: Non-steady states." The Review of Economic
    Studies 68.2 (2001): 323-346.

    Examples
    ---------

    >>> bm = pm.BidManager()
    >>> bm.add_bid(1, 3, 0)
    0
    >>> bm.add_bid(1, 0.5, 1)
    1
    >>> bm.add_bid(1, 1, 2, False)
    2
    >>> bm.add_bid(1, 2, 3, False)
    3
    >>> r = np.random.RandomState(420)
    >>> trans, extra = p2p_random(bm.get_df(), r=r)
    >>> extra
    {'trading_list': [[(0, 3), (1, 2)]]}
    >>> trans.get_df()
       bid  quantity  price  source  active
    0    0         1    2.5       3   False
    1    3         1    2.5       0   False
    2    1         0    0.0       2    True
    3    2         0    0.0       1    True

    """
    global p2p_mech_net_profit
    global p2p_mech_avg_profit

    p2p_mech_net_profit = 0
    p2p_mech_avg_profit = 0

    r = np.random.RandomState() if r is None else r
    trans = pm.TransactionManager()
    buying = bids[bids.buying]
    selling = bids[bids.buying == False]
    Nb, Ns = buying.shape[0], selling.shape[0]

    quantities = bids.quantity.values.copy()
    prices = bids.price.values.copy()

    inactive_buying = []
    inactive_selling = []

    # Enumerate all possible trades
    pairs = np.ones((Nb + Ns, Nb * Ns), dtype=bool)
    pairs_inv = []
    i = 0
    for b in buying.index:
        for s in selling.index:
            pairs[b, i] = False  # Row b has 0s whenever the pair involves b
            pairs[s, i] = False  # Same for s
            pairs_inv.append((b, s))
            i += 1

    active = np.ones(Nb * Ns, dtype=bool)
    tmp_active = active.copy()
    general_trading_list = []

    profit_sums = np.zeros(Nb + Ns)
    profit_div = np.ones(Nb + Ns)

    # Loop while there is quantities to trade or not all
    # possibilities have been tried

    while quantities.sum() > 0 and tmp_active.sum() > 0:
        trading_list = []
        while tmp_active.sum() > 0:  # We can select a pair
            where = np.where(tmp_active == 1)[0]
            x = r.choice(where)
            trade = pairs_inv[x]
            active[x] = False  # Pair has already traded
            trading_list.append(trade)
            tmp_active &= pairs[trade[0], :]  # buyer and seller already used
            tmp_active &= pairs[trade[1], :]

        general_trading_list.append(trading_list)
        for (b, s) in trading_list:
            buyer_price = prices[b] - wheelings[b][s];
            q = min(quantities[b], quantities[s])

            if q > 0 and buyer_price >= prices[s]:
                p = buyer_price * p_coef + (1 - p_coef) * prices[s]

                # p2p_mech_net_profit += ((buyer_price - p) * q) + ((p - prices[s]) * q)
                profit_sums[b] += (buyer_price - p) * q
                profit_sums[s] += (p - prices[s]) * q

                profit_div[b] += 1
                profit_div[s] += 1

                trans_b = (b, q, p, s, (quantities[b] - q) > 0)
                trans_s = (s, q, p, b, (quantities[s] - q) > 0)
                quantities[b] -= q
                quantities[s] -= q
            else:
                trans_b = (b, 0, 0, s, True)
                trans_s = (s, 0, 0, b, True)
            trans.add_transaction(*trans_b)
            trans.add_transaction(*trans_s)

        inactive_buying = [b for b in buying.index if quantities[b] == 0]
        inactive_selling = [s for s in selling.index if quantities[s] == 0]

        tmp_active = active.copy()
        for inactive in inactive_buying + inactive_selling:
            tmp_active &= pairs[inactive, :]

    p2p_mech_avg_profit = np.sum(np.divide(profit_sums, profit_div))
    p2p_mech_net_profit = np.sum(profit_sums)

    extra = {'trading_list': general_trading_list}
    return trans, extra


class P2PEGATTrading(pm.Mechanism):

    """Interface for P2PTrading.

    Parameters
    -----------

    bids: pd.DataFrame
        Collections of bids to use

    """

    def __init__(self, bids, *args, **kwargs):
        """
        """
        pm.Mechanism.__init__(self, p2p_egat_mechanism, bids, *args, **kwargs)


pm.market.MECHANISM['p2p-egat'] = P2PEGATTrading


def plot_network_diagram(
        bids,
        transactions,
        ax=None):
    """Plots all the bids as a bipartit graph
    with buyers and trades and an edge between
    each pair that traded

    Parameters
    ----------
    bids : pd.DataFrame
        Collection of bids to be used
    transactions : pd.DataFrame
        Collection of transactions to be used
    ax : pyplot.axe
        The axe in which the figure should be ploted

    Returns
    -------
     axe : matplotlib.axes._subplots.AxesSubplot
        The axe in which the figure was plotted.

    """
    bids = bids.get_df()
    tmp = transactions.get_df()
    #tmp = tmp.loc[(tmp['price']>0)]
    tmp['user_1'] = tmp.bid.map(bids.user)
    tmp['user_2'] = tmp.source.map(bids.user)
    #+1
    tmp['user_1'] = tmp['user_1'] + 1
    tmp['user_2'] = tmp['user_2'] + 1
    tmp['buying'] = tmp.bid.map(bids.buying)
    #test
    #tmp = tmp.loc[(tmp['price']>0)]
    buyers = bids.loc[bids['buying']].index.values
    #+1
    buyers = buyers + 1

    G = nx.from_pandas_edgelist(tmp, 'user_1', 'user_2')

    edge_labels = OrderedDict()
    duplicated_labels = tmp.set_index(
        ['user_1', 'user_2'])['quantity'].to_dict()
    for (x, y), v in duplicated_labels.items():
        if ((x, y) not in edge_labels and (y, x) not in edge_labels):
            edge_labels[(x, y)] = v

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    pos = nx.bipartite_layout(G, buyers, align='horizontal', scale=3)
    _ = nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_color='red',
        nodelist=[1,2,3],
        node_size=500)
    _ = nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_color='blue',
        nodelist=[4,5,6],
        node_size=500)
    _ = nx.draw_networkx_labels(G, pos=pos, ax=ax, font_color='w')
    _ = nx.draw_networkx_edges(G, pos=pos, label=G, ax=ax)
    _ = nx.draw_networkx_edge_labels(
        G,
        pos=pos,
        edge_labels=edge_labels,
        label_pos=0.8,
        ax=ax,
        font_size=6)
    _ = ax.axis('off')

    return ax

is_running = 'running' in st.session_state and st.session_state['running'] == True

if not is_running and 'run' not in st.session_state:
    # Initialize session state variables for 6 players
    # Player 1
    q_rand = random.randint(10,20)
    p_rand = random.uniform(3.65,3.75)
    if 'slider_q1' not in st.session_state:
        st.session_state['slider_q1'] = q_rand
    if 'num_q1' not in st.session_state:
        st.session_state['num_q1'] = q_rand
    if 'slider_p1' not in st.session_state:
        st.session_state['slider_p1'] = p_rand + 0.87
    if 'num_p1' not in st.session_state:
        st.session_state['num_p1'] = p_rand + 0.87
    # Player 2
    q_rand = random.randint(15,20)
    p_rand = random.uniform(3.70,3.80)
    if 'slider_q2' not in st.session_state:
        st.session_state['slider_q2'] = q_rand
    if 'num_q2' not in st.session_state:
        st.session_state['num_q2'] = q_rand
    if 'slider_p2' not in st.session_state:
        st.session_state['slider_p2'] = p_rand + 0.87
    if 'num_p2' not in st.session_state:
        st.session_state['num_p2'] = p_rand + 0.87
    #Player 3
    q_rand = random.randint(20,30)
    p_rand = random.uniform(3.75,3.85)
    if 'slider_q3' not in st.session_state:
        st.session_state['slider_q3'] = q_rand
    if 'num_q3' not in st.session_state:
        st.session_state['num_q3'] = q_rand
    if 'slider_p3' not in st.session_state:
        st.session_state['slider_p3'] = p_rand + 0.87
    if 'num_p3' not in st.session_state:
        st.session_state['num_p3'] = p_rand + 0.87
    #Player 4
    q_rand = random.randint(5,10)
    p_rand = random.uniform(3.65,3.75)
    if 'slider_q4' not in st.session_state:
        st.session_state['slider_q4'] = q_rand
    if 'num_q4' not in st.session_state:
        st.session_state['num_q4'] = q_rand
    if 'slider_p4' not in st.session_state:
        st.session_state['slider_p4'] = p_rand
    if 'num_p4' not in st.session_state:
        st.session_state['num_p4'] = p_rand
    #Player 5
    q_rand = random.randint(10,20)
    p_rand = random.uniform(3.70,3.80)
    if 'slider_q5' not in st.session_state:
        st.session_state['slider_q5'] = q_rand
    if 'num_q5' not in st.session_state:
        st.session_state['num_q5'] = q_rand
    if 'slider_p5' not in st.session_state:
        st.session_state['slider_p5'] = p_rand
    if 'num_p5' not in st.session_state:
        st.session_state['num_p5'] = p_rand
    #Player 6
    q_rand = random.randint(15,25)
    p_rand = random.uniform(3.75,3.85)
    if 'slider_q6' not in st.session_state:
        st.session_state['slider_q6'] = q_rand
    if 'num_q6' not in st.session_state:
        st.session_state['num_q6'] = q_rand
    if 'slider_p6' not in st.session_state:
        st.session_state['slider_p6'] = p_rand
    if 'num_p6' not in st.session_state:
        st.session_state['num_p6'] = p_rand

# Set Option for Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_option('theme.primaryColor', 'FFC726')
st.set_page_config(layout="wide", page_title='P2P Energy Market Simulation')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
# Disable Hamburger Icon
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Title of the App
st.title("EGAT P2P Energy Market: Matching Algorithm Simulation")

# On Change Sliders also change NumInputs
def update_slider():
    players = ['1', '2', '3', '4', '5', '6']
    for i in players:
        st.session_state['slider_q' + i] = st.session_state['num_q' + i]
        st.session_state['slider_p' + i] = st.session_state['num_p' + i]
# On Change NumInputs also change Sliders
def update_numin():
    players = ['1', '2', '3', '4', '5', '6']
    for i in players:
        st.session_state['num_q' + i] = st.session_state['slider_q' + i]
        st.session_state['num_p' + i] = st.session_state['slider_p' + i]
def update_slider1():
    players = ['1', '2', '3', '4', '5', '6']
    for i in players:
        st.session_state['slider_q' + i] = st.session_state['num_q' + i]
        st.session_state['slider_p' + i] = 4.62
def update_numin1():
    players = ['1', '2', '3', '4', '5', '6']
    for i in players:
        st.session_state['num_q' + i] = st.session_state['slider_q' + i]
        st.session_state['num_p' + i] = 4.62

# Recommended Price ***
recommended_price = 3.75

st.header(":zap: Bids/Offers")

st.markdown("**Simulated Wheeling Charges**")
df = pd.DataFrame(data={
    '': ['Player 4', 'Player 5', 'Player 6'],
    'Player 1': wheelings[0][3:],
    'Player 2': wheelings[1][3:],
    'Player 3': wheelings[2][3:],
})
st.dataframe(df.set_index(df.columns[0]))


colRunning, colButton1, colButton2 = st.columns([7,2,1])

with colButton1:
    timer_value = st.time_input('Run for at least (seconds)', datetime.time(0, 5), step=5*60, disabled=is_running)
    if timer_value.hour == 0 and timer_value.minute == 0:
        st.caption("00:00 means it will execute only 1 time")

def save_vars(to_save_vars: list[str]):
    for var in to_save_vars:
        if var in globals():
            st.session_state[var] = globals()[var]

def restore_vars(to_save_vars: list[str]):
    for var in to_save_vars:
        globals()[var] = st.session_state[var]

def has_all_vars(to_save_vars: list[str]):
    for var in to_save_vars:
        if var not in st.session_state:
            return False

    return True

def delete_all_vars(to_save_vars: list[str]):
    for var in to_save_vars:
        if var in st.session_state:
            del st.session_state[var]          

needSave = False
with colButton2:
    if st.button(":bar_chart: Run P2P Market Clearing", disabled=is_running):
        needSave = True


input_container = st.container();
with input_container:
    expanded = not is_running and 'run' not in st.session_state
    with st.expander("Input:", expanded=expanded):
        buyer_tab, seller_tab = st.tabs(['Buyer', 'Seller'])
        with buyer_tab:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(":one: **Player 1 :red[- Buyer: ]**")
                mode1 = st.radio("Select Operation Mode", ["Automatic", "Advanced"],horizontal=True,key='mode1', index=1, disabled=is_running)
                st.markdown('**:red[Quantity from actual]**')
                slide_q1 = st.slider('Select Quantity Bid (kWh):', 0, 30, key='slider_q1', step=1, on_change=update_numin, disabled=is_running)
                num_q1 = st.number_input('Enter Quantity Bid (kWh):',0, 30, key='num_q1', step=1, on_change=update_slider, disabled=is_running)
                if mode1 == 'Advanced':
                    st.markdown('**:red[Price included wheeling charge*]**')
                    slider_p1 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, 4.62, key='slider_p1', step=0.01, on_change=update_numin, disabled=is_running)
                    num_p1 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, 4.62, key='num_p1', step=0.01, on_change=update_slider, disabled=is_running)
                    st.caption("Hr-1 trade period price is 4.62 ฿/kWh")
                else:
                    st.markdown('**:red[Price included wheeling charge*]**')
                    st.markdown(':orange[Using Price Bid = 4.62 ฿/kWh]')
                    st.caption("Hr-1 trade period price is 4.62 ฿/kWh")
                    slider_p1 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, value = 4.62, key='slider_p1', step=0.01, on_change=update_numin1, disabled=True) 
                    num_p1 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, value = 4.62, key='num_p1', step=0.01, on_change=update_slider1, disabled=True)
                    num_p1 = 4.62
                q1 = num_q1
                p1 = num_p1
                st.markdown("---")
                st.markdown("Bid Details for Player 1:")
                st.markdown("Quantity = " + ":red[" + "{:.2f}".format(q1) + "] " + "kWh")
                st.markdown("Price* = " + ":red[" + "{:.2f}".format(p1) + "] " + "฿/kWh")
                #st.caption(f':red_circle: Price for matching will be {p1-0.87:.2f} ฿/kWh')
                st.caption('*Included wheeling charge')
            with col2:
                st.markdown(":two: **Player 2 :red[- Buyer: ]**")
                mode2 = st.radio("Select Operation Mode", ["Automatic", "Advanced"],horizontal=True,key='mode2', index=1, disabled=is_running)
                st.markdown('**:red[Quantity from actual]**')
                slide_q2 = st.slider('Select Quantity Bid (kWh):', 0, 30, key='slider_q2', step=1, on_change=update_numin, disabled=is_running)
                num_q2 = st.number_input('Enter Quantity Bid (kWh):',0, 30, key='num_q2', step=1, on_change=update_slider, disabled=is_running)
                if mode2 == 'Advanced':
                    st.markdown('**:red[Price included wheeling charge*]**')
                    slider_p2 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, 4.62, key='slider_p2', step=0.01, on_change=update_numin, disabled=is_running)
                    num_p2 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, 4.62, key='num_p2', step=0.01, on_change=update_slider, disabled=is_running)
                    st.caption("Hr-1 trade period price is 4.62 ฿/kWh")
                else:
                    st.markdown('**:red[Price included wheeling charge*]**')
                    st.markdown(':orange[Using Price Bid = 4.62 ฿/kWh]')
                    st.caption("Hr-1 trade period price is 4.62 ฿/kWh")
                    slider_p2 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, value = 4.62, key='slider_p2', step=0.01, on_change=update_numin1, disabled=True) 
                    num_p2 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, value = 4.62, key='num_p2', step=0.01, on_change=update_slider1, disabled=True)
                    num_p2 = 4.62   
                q2 = num_q2
                p2 = num_p2
                st.markdown("---")
                st.markdown("Bid Details for Player 2:")
                st.markdown("Quantity = " + ":red[" + "{:.2f}".format(q2) + "] " + "kWh")
                st.markdown("Price* = " + ":red[" + "{:.2f}".format(p2) + "] " + "฿/kWh")
                #st.caption(f':red_circle: Price for matching will be {p2-0.87:.2f} ฿/kWh')
                st.caption('*Included wheeling charge')
            with col3:
                st.markdown(":three: **Player 3 :red[- Buyer: ]**")
                mode3 = st.radio("Select Operation Mode", ["Automatic", "Advanced"],horizontal=True,key='mode3', index=1, disabled=is_running)
                st.markdown('**:red[Quantity from actual]**')
                slide_q3 = st.slider('Select Quantity Bid (kWh):', 0, 30, key='slider_q3', step=1, on_change=update_numin, disabled=is_running)
                num_q3 = st.number_input('Enter Quantity Bid (kWh):',0, 30, key='num_q3', step=1, on_change=update_slider, disabled=is_running)
                if mode3 == 'Advanced':
                    st.markdown('**:red[Price included wheeling charge*]**')
                    slider_p3 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, 4.62, key='slider_p3', step=0.01, on_change=update_numin, disabled=is_running)
                    num_p3 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, 4.62, key='num_p3', step=0.01, on_change=update_slider, disabled=is_running)
                    st.caption("Hr-1 trade period price is 4.62 ฿/kWh")
                else:
                    st.markdown('**:red[Price included wheeling charge*]**')
                    st.markdown(':orange[Using Price Bid = 4.62 ฿/kWh]')
                    st.caption("Hr-1 trade period price is 4.62 ฿/kWh")
                    slider_p3 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, value = 4.62, key='slider_p3', step=0.01, on_change=update_numin1, disabled=True) 
                    num_p3 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, value = 4.62, key='num_p3', step=0.01, on_change=update_slider1, disabled=True)
                    num_p3 = 4.62
                q3 = num_q3
                p3 = num_p3
                st.markdown("---")
                st.markdown("Bid Details for Player 3:")
                st.markdown("Quantity = " + ":red[" + "{:.2f}".format(q3) + "] " + "kWh")
                st.markdown("Price* = " + ":red[" + "{:.2f}".format(p3) + "] " + "฿/kWh")
                #st.caption(f':red_circle: Price for matching will be {p3-0.87:.2f} ฿/kWh')
                st.caption('*Included wheeling charge')

        with seller_tab:
            col4, col5, col6 = st.columns(3)
            with col4:
                st.markdown(":four: **Player 4 :blue[- Seller: ]**")
                mode4 = st.radio("Select Operation Mode", ["Automatic", "Advanced"],horizontal=True,key='mode4', index=1, disabled=is_running)
                st.markdown('**:blue[Quantity from actual]**')
                slide_q4 = st.slider('Select Quantity Bid (kWh):', 0, 30, key='slider_q4', step=1, on_change=update_numin, disabled=is_running)
                num_q4 = st.number_input('Enter Quantity Bid (kWh):',0, 30, key='num_q4', step=1, on_change=update_slider, disabled=is_running)
                if mode4 == 'Advanced':
                    st.markdown('**:blue[Price]**')
                    slider_p4 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, 3.75, key='slider_p4', step=0.01, on_change=update_numin, disabled=is_running)
                    num_p4 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, 3.75, key='num_p4', step=0.01, on_change=update_slider, disabled=is_running)
                    st.caption("Hr-1 trade period price is 3.75 ฿/kWh")
                else:
                    st.markdown('**:blue[Price]**')
                    st.markdown(':orange[Using Price Bid = 3.75 ฿/kWh]')
                    st.caption("Hr-1 trade period price is 3.75 ฿/kWh")
                    slider_p4 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, value = 4.62, key='slider_p4', step=0.01, on_change=update_numin1, disabled=True) 
                    num_p4 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, value = 4.62, key='num_p4', step=0.01, on_change=update_slider1, disabled=True)
                    num_p4 = 3.75
                q4 = num_q4
                p4 = num_p4
                st.markdown("---")
                st.markdown("Offer Details for Player 4:")
                st.markdown("Quantity = " + ":blue[" + "{:.2f}".format(q4) + "] " + "kWh")
                st.markdown("Price = " + ":blue[" + "{:.2f}".format(p4) + "] " + "฿/kWh")
                st.caption(f':large_blue_circle: Price for matching will be {p4:.2f} ฿/kWh')
            with col5:
                st.markdown(":five: **Player 5 :blue[- Seller: ]**")
                mode5 = st.radio("Select Operation Mode", ["Automatic", "Advanced"],horizontal=True,key='mode5', index=1)
                st.markdown('**:blue[Quantity from actual]**')
                slide_q5 = st.slider('Select Quantity Bid (kWh):', 0, 30, key='slider_q5', step=1, on_change=update_numin, disabled=is_running)
                num_q5 = st.number_input('Enter Quantity Bid (kWh):',0, 30, key='num_q5', step=1, on_change=update_slider, disabled=is_running)
                if mode5 == 'Advanced':
                    st.markdown('**:blue[Price]**')
                    slider_p5 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, 3.75, key='slider_p5', step=0.01, on_change=update_numin, disabled=is_running)
                    num_p5 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, 3.75, key='num_p5', step=0.01, on_change=update_slider, disabled=is_running)
                    st.caption("Hr-1 trade period price is 3.75 ฿/kWh")
                else:
                    st.markdown('**:blue[Price]**')
                    st.markdown(':orange[Using Price Bid = 3.75 ฿/kWh]')
                    st.caption("Hr-1 trade period price is 3.75 ฿/kWh")
                    slider_p5 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, value = 4.62, key='slider_p5', step=0.01, on_change=update_numin1, disabled=True) 
                    num_p5 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, value = 4.62, key='num_p5', step=0.01, on_change=update_slider1, disabled=True)
                    num_p5 = 3.75
                q5 = num_q5
                p5 = num_p5
                st.markdown("---")
                st.markdown("Offer Details for Player 5:")
                st.markdown("Quantity = " + ":blue[" + "{:.2f}".format(q5) + "] " + "kWh")
                st.markdown("Price = " + ":blue[" + "{:.2f}".format(p5) + "] " + "฿/kWh")
                st.caption(f':large_blue_circle: Price for matching will be {p4:.2f} ฿/kWh')
            with col6:
                st.markdown(":six: **Player 6 :blue[- Seller: ]**")
                mode6 = st.radio("Select Operation Mode", ["Automatic", "Advanced"],horizontal=True,key='mode6', index=1)
                st.markdown('**:blue[Quantity from actual]**')
                slide_q6 = st.slider('Select Quantity Bid (kWh):', 0, 30, key='slider_q6', step=1, on_change=update_numin, disabled=is_running)
                num_q6 = st.number_input('Enter Quantity Bid (kWh):',0, 30, key='num_q6', step=1, on_change=update_slider, disabled=is_running)
                if mode6 == 'Advanced':
                    st.markdown('**:blue[Price]**')
                    slider_p6 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, 3.75, key='slider_p6', step=0.01, on_change=update_numin, disabled=is_running)
                    num_p6 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, 3.75, key='num_p6', step=0.01, on_change=update_slider, disabled=is_running)
                    st.caption("Hr-1 trade period price is 3.75 ฿/kWh")
                else:
                    st.markdown('**:blue[Price]**')
                    st.markdown(':orange[Using Price Bid = 3.75 ฿/kWh]')
                    st.caption("Hr-1 trade period price is 3.75 ฿/kWh")
                    slider_p6 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, value = 4.62, key='slider_p6', step=0.01, on_change=update_numin1, disabled=True) 
                    num_p6 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, value = 4.62, key='num_p6', step=0.01, on_change=update_slider1, disabled=True)
                    num_p6 = 3.75
                q6 = num_q6
                p6 = num_p6
                st.markdown("---")
                st.markdown("Offer Details for Player 6:")
                st.markdown("Quantity = " + ":blue[" + "{:.2f}".format(q6) + "] " + "kWh")
                st.markdown("Price = " + ":blue[" + "{:.2f}".format(p6) + "] " + "฿/kWh")
                st.caption(f':large_blue_circle: Price for matching will be {p4:.2f} ฿/kWh')
        
but1,but2,but3 = st.columns(3)

if needSave:
    st.session_state['run'] = False
    st.session_state['running'] = True
    st.session_state['running_start'] = datetime.datetime.now()
    st.session_state['running_end'] = datetime.datetime.now() + datetime.timedelta(minutes=timer_value.hour,seconds=timer_value.minute)
    is_running = True
    save_vars([
        'p1', 'q1', 'p2', 'q2', 'p3', 'q3', 
        'p4', 'q4', 'p5', 'q5', 'p6', 'q6'
    ])

    sum_q_bid = sum([q1,q2,q3])
    avg_bid = sum([p1,p2,p3])/3
    save_vars(['sum_q_bid', 'avg_bid'])

    sum_q_offer = sum([q4,q5,q6])
    avg_offer = sum([p4,p5,p6])/3
    save_vars(['sum_q_offer', 'avg_offer'])

    delete_all_vars(['mar', 'bids', 'transactions', 'extras', 'current_max_net_profit'])

    if not (timer_value.hour == 0 and timer_value.minute == 0 and timer_value.second == 0): 
        input_container.empty()     
        st.rerun()
        

current_max_net_profit = 0
current_max_avg_profit = 0

with colRunning:
    run_counter = 0
    if not is_running:
        with st.empty():
            colNetProfit, colAvgProfit, colTimer, colTotalCount, _ = st.columns([2,2,2,2,2])
            with colTimer:
                if has_all_vars(['running_start', 'running_actual_end']):
                    current_time: datetime.datetime = st.session_state['running_start']
                    target_time: datetime.datetime = st.session_state['running_actual_end']
                    time_delta = target_time - current_time

                    minutes = int(time_delta.total_seconds() / (60))
                    seconds = int(time_delta.total_seconds() % 60)
                    milliseconds = int((time_delta.total_seconds() * 1000) % 1000)

                    st.metric("Run Time", f'{minutes:02d}:{seconds:02d}.{milliseconds:03d}')

            with colNetProfit:
                if 'run' in st.session_state:
                    restore_vars(['current_max_net_profit'])
                    st.metric("Highest\nTotal Profits Optimization", f'{current_max_net_profit:.2f} ฿')

            with colAvgProfit:
                if 'run' in st.session_state:
                    restore_vars(['current_max_avg_profit'])
                    st.metric("Highest\nAverage Profits Optimization", f'{current_max_avg_profit:.2f} ฿')

            with colTotalCount:
                if has_all_vars(['run_counter']):
                    restore_vars([
                        'run_counter'
                    ])

                    st.metric("Total Run", f'{run_counter}')


    if is_running:
        with st.empty():
            while is_running:
                colNetProfit, colAvgProfit, colTimer, colTotalCount, _ = st.columns([2,2,2,2,2])

                restore_vars([
                    'p1', 'q1', 'p2', 'q2', 'p3', 'q3', 
                    'p4', 'q4', 'p5', 'q5', 'p6', 'q6'
                ])

                with colTimer:
                    mar = pm.Market()
                    mar.accept_bid(q1, round(p1,2), 0, True, 0, True)
                    mar.accept_bid(q2, round(p2,2), 1, True, 0, True)
                    mar.accept_bid(q3, round(p3,2), 2, True, 0, True)
                    mar.accept_bid(q4, round(p4,2), 3, False, 0, True)
                    mar.accept_bid(q5, round(p5,2), 4, False, 0, True)
                    mar.accept_bid(q6, round(p6,2), 5, False, 0, True)
                    bids = mar.bm.get_df()
                    transactions, extras = mar.run('p2p-egat') # run the p2p mechanism

                    if run_counter == 0:
                        current_max_net_profit = p2p_mech_net_profit
                        current_max_avg_profit = p2p_mech_avg_profit
                        save_vars(['mar', 'bids', 'transactions', 'extras', 'current_max_net_profit', 'current_max_avg_profit'])
                    else:
                        if ((current_max_net_profit < p2p_mech_net_profit) 
                            or (current_max_net_profit == p2p_mech_net_profit 
                                and current_max_avg_profit > p2p_mech_avg_profit)
                        ):
                            current_max_net_profit = p2p_mech_net_profit
                            current_max_avg_profit = p2p_mech_avg_profit
                            save_vars(['mar', 'bids', 'transactions', 'extras', 'current_max_net_profit', 'current_max_avg_profit'])
                        else:
                            if has_all_vars(['mar', 'bids', 'transactions', 'extras']):
                                restore_vars(['mar', 'bids', 'transactions', 'extras'])

                    current_time: datetime.datetime = datetime.datetime.now()
                    target_time: datetime.datetime = st.session_state['running_end']
                    time_delta = target_time - current_time

                    minutes = int(time_delta.total_seconds() / (60))
                    seconds = int(time_delta.total_seconds())

                    if time_delta.total_seconds() <= 0:
                        st.session_state['run'] = True
                        st.session_state['running'] = False

                        is_running = False

                        st.session_state['running_actual_end'] = datetime.datetime.now()

                        # if run_counter > 0:
                        st.rerun()

                        break
                    else:
                        st.metric("Running", f'{minutes:02d}:{seconds:02d}')


                with colNetProfit:
                    if not is_running:
                        st.metric("Highest\nTotal Profit Optimization", f'{current_max_net_profit:.2f} ฿')
                    else:
                        st.metric("Current Highest\nTotal Profit Optimization", f'{current_max_net_profit:.2f} ฿')

                with colAvgProfit:
                    if not is_running:
                        st.metric("Highest\nAverage Profits Optimization", f'{current_max_avg_profit:.2f} ฿')
                    else:
                        st.metric("Current\nAverage Profits Optimization", f'{current_max_avg_profit:.2f} ฿')

                with colTotalCount:
                    st.metric("Total Run", f'{run_counter}')

                run_counter += 1
                save_vars(['run_counter'])



if 'run' in st.session_state and not is_running:
    restore_vars([
        'p1', 'q1', 'p2', 'q1', 'p3', 'q3', 
        'p4', 'q4', 'p5', 'q5', 'p6', 'q6'
    ])
    restore_vars(['sum_q_bid', 'avg_bid'])
    restore_vars(['sum_q_offer', 'avg_offer'])

    st.success('Optimization completed.', icon="✅")
    colMatching1, colMatching2, colMatching3, res4 = st.columns([1.5,1.5,5,2])
    with colMatching1:
        st.markdown(':red[Demand bids for matching:]')
        st.caption(f'Sum bid quantity = {sum_q_bid}')
        st.caption(f'Average bid quantity = {sum_q_bid/3:.2f}')
        st.caption(f'Average bid price = {avg_bid:.2f}')
        #st.metric(label="Player 1:", value=f"{p1} ฿", delta=f"{(p1-avg_bid):.2f} ฿")
        st.code(f'''Player 1: 
        Quantity = {q1}
        Price* = {p1:.2f}''')
        st.code(f'''Player 2: 
        Quantity = {q2}
        Price* = {p2:.2f}''')
        st.code(f'''Player 3: 
        Quantity = {q3}
        Price* = {p3:.2f}''')
        st.caption('Before matching:')
        st.caption('Included wheeling charge')
    with colMatching2:
        st.markdown(':blue[Supply offers for matching:]')
        st.caption(f'Sum offer quantity = {sum_q_offer}')
        st.caption(f'Average offer quantity = {sum_q_offer/3:.2f}')
        st.caption(f'Average offer price = {avg_offer:.2f}')
        #st.metric(label="Player 1:", value=f"{p1} ฿", delta=f"{(p1-avg_bid):.2f} ฿")
        st.code(f'''Player 4:
        Quantity = {q4}
        Price = {p4:.2f}''')
        st.code(f'''Player 5:
        Quantity = {q5}
        Price = {p5:.2f}''')
        st.code(f'''Player 6:
        Quantity = {q6}
        Price = {p6:.2f}''')
    with colMatching3:
        restore_vars(['mar', 'bids', 'transactions', 'extras'])

        st.markdown(':orange[Market Clearing Chart:]')
        t1, t2  = st.tabs(['Network Diagram', 'Demand/Supply Curve'])
        with t1:
            st.pyplot(plot_network_diagram(mar.bm,transactions).figure)
        with t2:    
            #st.pyplot(pm.plot.trades.plot_trades_as_graph(mar.bm,transactions).figure)
            st.pyplot(pm.plot.demand_curves.plot_demand_curves(mar.bm.get_df(),None,1,1.2))
    with res4:
        st.markdown(':green[Transactions: ]')
        st.caption('Matched price excluded wheeling charge')
        df = transactions.get_df()
        df['userA'] = df['bid'] + 1
        df['userB'] = df['source'] + 1
        df['price'] = df['price'].replace(0, np.nan)
        # Add round to format 2 decimal places
        df['price'] = df['price'].round(2)
        st.table(df.loc[:,['userA','userB', 'quantity', 'price']].sort_values(by='userA').style.format({'price': '{:.2f}'}, na_rep='-'))
        st.latex('P_{p2p} = 0.5*P_{buyer} + 0.5*P_{seller}') 
        st.text(f"Minimum cleared price = {df['price'].min():.2f} ฿/kWh")
        st.text(f"Maximum cleared price = {df['price'].max():.2f} ฿/kWh")
        # st.write('Percentage of the maximum possible traded quantity')
        # st.write(f"{stats['percentage_traded']}%")
        # st.write('Percentage of the maximum possible total welfare')
        # st.write(f"{stats['percentage_welfare']}%")
        # st.write('Profits per user')
        # for u in bids.user.unique():
        #     st.write(f'User {u:2} obtained a profit of {stats["profits"]["player_bid"][u]:0.2f}')
        # st.write(f'Profit to Market Maker was {stats["profits"]["market"]:0.2f}')
    with st.expander('Summarize Trading Result:', True):
        #st.markdown('test')
        #st.write(stats)
        # for index, row in df.sort_values(by='userA').iterrows():
        #     st.write((f"Index: {index} | userA: {row['userA']} | userB: {row['userB']} | quantity: {row['quantity']} | price: {row['price']}"))
        # players = [':one:',':two:',':three:',':four:',':five:',':six:']
        # for row in df.sort_values(by='userA').iterrows():
        #     #icon 
        #     st.markdown(f"{players[int(row['userA'])]} :white_check_mark")
        df_result = df.sort_values(by='userA')
        colResult1, colResult2, colResult3, colResult4, colResult5, colResult6 = st.columns(6)
        with colResult1:
            st.markdown(":one: **Player 1 :red[- Buyer: ]**")
            st.markdown('• A.I. Automatic Matching:')
            st.code(f'''Bid: 
            Quantity = {q1}
            Price* = {p1:.2f}''')
            if df_result.loc[df['userA'] == 1]['quantity'].sum(axis=0) == 0:
                st.write(f"**:x: Didn't match with anyone.**")
                st.markdown('---')
                st.markdown(f'**Quantity did not match:**')
                st.markdown(f':warning: Buy from grid :orange[{q1} kWh]')
            else:
                cnt = 1
                for index, row in df_result.loc[df['userA'] == 1].iterrows():
                    if row['quantity'] > 0:
                        st.markdown(f"**Matched pair {cnt}:**")
                        st.metric("Matched quantity", "{:d} kWh".format(row['quantity']))
                        st.metric("Matched price* with profit","{:.2f} ฿/kWh".format(row['price']),"{:.2f} ฿/kWh discount".format(row['price']-(p1-wheelings[0][row['userB']-1])),"inverse","Matched price = Pbid + Psharing_profit")
                        st.caption(f"{row['price']+wheelings[0][row['userB']-1]:.2f} ฿/kWh including wheeling charge {wheelings[0][row['userB']-1]:.2f} ฿/kWh")
                        st.markdown(f":white_check_mark: Matched with Player {row['userB']}")
                        st.markdown(f"Pay :violet[{(row['price']*row['quantity']):.2f} ฿] to Player {row['userB']}")
                        st.markdown(f"Pay :violet[{wheelings[0][row['userB']-1]*row['quantity']:.2f} ฿] to TSO/DSO")
                        st.markdown('---')
                        cnt += 1
                if q1 != df_result.loc[df['userA'] == 1]['quantity'].sum(axis=0):
                    q1_buy_from_grid = q1 - df_result.loc[df['userA'] == 1]['quantity'].sum(axis=0)
                    st.markdown(f'**Quantity did not match:**')
                    st.markdown(f':warning: Buy from grid :orange[{q1_buy_from_grid} kWh]')

        with colResult2:
            st.markdown(":two: **Player 2 :red[- Buyer: ]**")
            st.markdown('• A.I. Automatic Matching:')
            st.code(f'''Bid: 
            Quantity = {q2}
            Price* = {p2:.2f}''')
            if df_result.loc[df['userA'] == 2]['quantity'].sum(axis=0) == 0:
                st.write(f"**:x: Didn't match with anyone.**")
                st.markdown('---')
                st.markdown(f'**Quantity did not match:**')
                st.markdown(f':warning: Buy from grid :orange[{q2} kWh]')
            else:
                cnt = 1
                for index, row in df_result.loc[df['userA'] == 2].iterrows():
                    if row['quantity'] > 0:
                        st.markdown(f"**Matched pair {cnt}:**")
                        st.metric("Matched quantity", "{:d} kWh".format(row['quantity']))
                        st.metric("Matched price* with profit","{:.2f} ฿/kWh".format(row['price']),"{:.2f} ฿/kWh discount".format(row['price']-(p2-wheelings[1][row['userB']-1])),"inverse","Matched price = Pbid + Psharing_profit")
                        st.caption(f"{row['price']+wheelings[1][row['userB']-1]:.2f} ฿/kWh including wheeling charge {wheelings[1][row['userB']-1]:.2f} ฿/kWh")
                        st.markdown(f":white_check_mark: Matched with Player {row['userB']}")
                        st.markdown(f"Pay :violet[{(row['price']*row['quantity']):.2f} ฿] to Player {row['userB']}")
                        st.markdown(f"Pay :violet[{wheelings[1][row['userB']-1]*row['quantity']:.2f} ฿] to TSO/DSO")
                        st.markdown('---')
                        cnt += 1
                if q2 != df_result.loc[df['userA'] == 2]['quantity'].sum(axis=0):
                    q2_buy_from_grid = q2 - df_result.loc[df['userA'] == 2]['quantity'].sum(axis=0)
                    st.markdown(f'**Quantity did not match:**')
                    st.markdown(f':warning: Buy from grid :orange[{q2_buy_from_grid} kWh]')
        with colResult3:
            st.markdown(":three: **Player 3 :red[- Buyer: ]**")
            st.markdown('• A.I. Automatic Matching:')
            st.code(f'''Bid: 
            Quantity = {q3}
            Price* = {p3:.2f}''')
            if df_result.loc[df['userA'] == 3]['quantity'].sum(axis=0) == 0:
                st.write(f"**:x: Didn't match with anyone.**")
                st.markdown('---')
                st.markdown(f'**Quantity did not match:**')
                st.markdown(f':warning: Buy from grid :orange[{q3} kWh]')
            else:
                cnt = 1
                for index, row in df_result.loc[df['userA'] == 3].iterrows():
                    if row['quantity'] > 0:
                        st.markdown(f"**Matched pair {cnt}:**")
                        st.metric("Matched quantity", "{:d} kWh".format(row['quantity']))
                        st.metric("Matched price* with profit","{:.2f} ฿/kWh".format(row['price']),"{:.2f} ฿/kWh discount".format(row['price']-(p3-wheelings[2][row['userB']-1])),"inverse","Matched price = Pbid + Psharing_profit")
                        st.caption(f"{row['price']+wheelings[2][row['userB']-1]:.2f} ฿/kWh including wheeling charge {wheelings[2][row['userB']-1]:.2f} ฿/kWh")
                        st.markdown(f":white_check_mark: Matched with Player {row['userB']}")
                        st.markdown(f"Pay :violet[{(row['price']*row['quantity']):.2f} ฿] to Player {row['userB']}")
                        st.markdown(f"Pay :violet[{wheelings[2][row['userB']-1]*row['quantity']:.2f} ฿] to TSO/DSO")
                        st.markdown('---')
                        cnt += 1
                if q3 != df_result.loc[df['userA'] == 3]['quantity'].sum(axis=0):
                    q3_buy_from_grid = q3 - df_result.loc[df['userA'] == 3]['quantity'].sum(axis=0)
                    st.markdown(f'**Quantity did not match:**')
                    st.markdown(f':warning: Buy from grid :orange[{q3_buy_from_grid} kWh]')
        with colResult4:
            st.markdown(":four: **Player 4 :blue[- Seller: ]**")
            st.markdown('• A.I. Automatic Matching:')
            st.code(f'''Offer: 
            Quantity = {q4}
            Price = {p4:.2f}''')
            if df_result.loc[df['userA'] == 4]['quantity'].sum(axis=0) == 0:
                st.write(f"**:x: Didn't match with anyone.**")
            else:
                cnt = 1
                for index, row in df_result.loc[df['userA'] == 4].iterrows():
                    if row['quantity'] > 0:
                        st.markdown(f"**Matched pair {cnt}:**")
                        st.metric("Matched quantity", "{:d} kWh".format(row['quantity']))
                        st.metric("Matched price with profit","{:.2f} ฿/kWh".format(row['price']),"{:.2f} ฿/kWh profit".format(row['price']-(p4)),"normal","Matched price = Pbid + Psharing_profit")
                        st.caption('<font color="grey">–<br/>-</font>', unsafe_allow_html=True)
                        st.markdown(f":white_check_mark: Matched with Player {row['userB']}")
                        st.markdown(f"Get :violet[{(row['price']*row['quantity']):.2f} ฿] from Player {row['userB']}")
                        st.markdown('<font color="grey">–</font>', unsafe_allow_html=True)
                        st.markdown('---')
                        cnt += 1
                if q4 != df_result.loc[df['userA'] == 4]['quantity'].sum(axis=0):
                    q4_excess_to_grid = q4 - df_result.loc[df['userA'] == 4]['quantity'].sum(axis=0)
                    st.markdown(f'**Quantity did not match:**')
                    st.markdown(f':warning: Excess to grid :orange[{q4_excess_to_grid} kWh]')
        with colResult5:
            st.markdown(":five: **Player 5 :blue[- Seller: ]**")
            st.markdown('• A.I. Automatic Matching:')
            st.code(f'''Offer: 
            Quantity = {q5}
            Price = {p5:.2f}''')
            if df_result.loc[df['userA'] == 5]['quantity'].sum(axis=0) == 0:
                st.write(f"**:x: Didn't match with anyone.**")
            else:
                cnt = 1
                for index, row in df_result.loc[df['userA'] == 5].iterrows():
                    if row['quantity'] > 0:
                        st.markdown(f"**Matched pair {cnt}:**")
                        st.metric("Matched quantity", "{:d} kWh".format(row['quantity']))
                        st.metric("Matched price with profit","{:.2f} ฿/kWh".format(row['price']),"{:.2f} ฿/kWh profit".format(row['price']-(p5)),"normal","Matched price = Pbid + Psharing_profit")
                        st.caption('<font color="grey">–<br/>-</font>', unsafe_allow_html=True)
                        st.markdown(f":white_check_mark: Matched with Player {row['userB']}")
                        st.markdown(f"Get :violet[{(row['price']*row['quantity']):.2f} ฿] from Player {row['userB']}")
                        st.markdown('<font color="grey">–</font>', unsafe_allow_html=True)
                        st.markdown('---')
                        cnt += 1
                if q5 != df_result.loc[df['userA'] == 5]['quantity'].sum(axis=0):
                    q5_excess_to_grid = q5 - df_result.loc[df['userA'] == 5]['quantity'].sum(axis=0)
                    st.markdown(f'**Quantity did not match:**')
                    st.markdown(f':warning: Excess to grid :orange[{q5_excess_to_grid} kWh]')
        with colResult6:
            st.markdown(":six: **Player 6 :blue[- Seller: ]**")
            st.markdown('• A.I. Automatic Matching:')
            st.code(f'''Offer: 
            Quantity = {q6}
            Price = {p6:.2f}''')
            if df_result.loc[df['userA'] == 6]['quantity'].sum(axis=0) == 0:
                st.write(f"**:x: Didn't match with anyone.**")
            else:
                cnt = 1
                for index, row in df_result.loc[df['userA'] == 6].iterrows():
                    if row['quantity'] > 0:
                        st.markdown(f"**Matched pair {cnt}:**")
                        st.metric("Matched quantity", "{:d} kWh".format(row['quantity']))
                        st.metric("Matched price with profit","{:.2f} ฿/kWh".format(row['price']),"{:.2f} ฿/kWh profit".format(row['price']-(p6)),"normal","Matched price = Pbid + Psharing_profit")
                        st.caption('<font color="grey">–<br/>-</font>', unsafe_allow_html=True)
                        st.markdown(f":white_check_mark: Matched with Player {row['userB']}")
                        st.markdown(f"Get :violet[{(row['price']*row['quantity']):.2f} ฿] from Player {row['userB']}")
                        st.markdown('<font color="grey">–</font>', unsafe_allow_html=True)
                        st.markdown('---')
                        cnt += 1
                if q6 != df_result.loc[df['userA'] == 6]['quantity'].sum(axis=0):
                    q6_excess_to_grid = q6 - df_result.loc[df['userA'] == 6]['quantity'].sum(axis=0)
                    st.markdown(f'**Quantity did not match:**')
                    st.markdown(f':warning: Excess to grid :orange[{q6_excess_to_grid} kWh]')
else:
    st.info('Edit bids/offers before press button.', icon="ℹ️")

st.write('---')
st.markdown(':gray[This project is for demo purpose only. [Torsak]]')
