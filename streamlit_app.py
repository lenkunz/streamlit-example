import streamlit as st
import pandas as pd
import random
import pymarket as pm
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import pprint


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
with st.expander("Input:", True):
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(":one: **Player 1 :red[- Buyer: ]**")
        mode1 = st.radio("Select Operation Mode", ["Automatic", "Advanced"],horizontal=True,key='mode1', index=1)
        st.markdown('**:red[Quantity from actual]**')
        slide_q1 = st.slider('Select Quantity Bid (kWh):', 0, 30, key='slider_q1', step=1, on_change=update_numin)
        num_q1 = st.number_input('Enter Quantity Bid (kWh):',0, 30, key='num_q1', step=1, on_change=update_slider)
        if mode1 == 'Advanced':
            st.markdown('**:red[Price included wheeling charge*]**')
            slider_p1 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, 4.62, key='slider_p1', step=0.01, on_change=update_numin)
            num_p1 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, 4.62, key='num_p1', step=0.01, on_change=update_slider)
            st.caption("N-1 trade period price is 4.62 ฿/kWh")
        else:
            st.markdown('**:red[Price included wheeling charge*]**')
            st.markdown(':orange[Using Price Bid = 4.62 ฿/kWh]')
            st.caption("N-1 trade period price is 4.62 ฿/kWh")
            slider_p1 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, value = 4.62, key='slider_p1', step=0.01, on_change=update_numin1, disabled=True) 
            num_p1 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, value = 4.62, key='num_p1', step=0.01, on_change=update_slider1, disabled=True)
            num_p1 = 4.62
        q1 = num_q1
        p1 = num_p1
        st.markdown("---")
        st.markdown("Bid Details for Player 1:")
        st.markdown("Quantity = " + ":red[" + "{:.2f}".format(q1) + "] " + "kWh")
        st.markdown("Price* = " + ":red[" + "{:.2f}".format(p1) + "] " + "฿/kWh")
        st.caption(f':red_circle: Price for matching will be {p1-0.87:.2f} ฿/kWh')
        st.caption('*Included wheeling charge 0.87 ฿/kWh')
    with col2:
        st.markdown(":two: **Player 2 :red[- Buyer: ]**")
        mode2 = st.radio("Select Operation Mode", ["Automatic", "Advanced"],horizontal=True,key='mode2', index=1)
        st.markdown('**:red[Quantity from actual]**')
        slide_q2 = st.slider('Select Quantity Bid (kWh):', 0, 30, key='slider_q2', step=1, on_change=update_numin)
        num_q2 = st.number_input('Enter Quantity Bid (kWh):',0, 30, key='num_q2', step=1, on_change=update_slider)
        if mode2 == 'Advanced':
            st.markdown('**:red[Price included wheeling charge*]**')
            slider_p2 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, 4.62, key='slider_p2', step=0.01, on_change=update_numin)
            num_p2 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, 4.62, key='num_p2', step=0.01, on_change=update_slider)
            st.caption("N-1 trade period price is 4.62 ฿/kWh")
        else:
            st.markdown('**:red[Price included wheeling charge*]**')
            st.markdown(':orange[Using Price Bid = 4.62 ฿/kWh]')
            st.caption("N-1 trade period price is 4.62 ฿/kWh")
            slider_p2 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, value = 4.62, key='slider_p2', step=0.01, on_change=update_numin1, disabled=True) 
            num_p2 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, value = 4.62, key='num_p2', step=0.01, on_change=update_slider1, disabled=True)
            num_p2 = 4.62   
        q2 = num_q2
        p2 = num_p2
        st.markdown("---")
        st.markdown("Bid Details for Player 2:")
        st.markdown("Quantity = " + ":red[" + "{:.2f}".format(q2) + "] " + "kWh")
        st.markdown("Price* = " + ":red[" + "{:.2f}".format(p2) + "] " + "฿/kWh")
        st.caption(f':red_circle: Price for matching will be {p2-0.87:.2f} ฿/kWh')
        st.caption('*Included wheeling charge 0.87 THB (฿)')
    with col3:
        st.markdown(":three: **Player 3 :red[- Buyer: ]**")
        mode3 = st.radio("Select Operation Mode", ["Automatic", "Advanced"],horizontal=True,key='mode3', index=1)
        st.markdown('**:red[Quantity from actual]**')
        slide_q3 = st.slider('Select Quantity Bid (kWh):', 0, 30, key='slider_q3', step=1, on_change=update_numin)
        num_q3 = st.number_input('Enter Quantity Bid (kWh):',0, 30, key='num_q3', step=1, on_change=update_slider)
        if mode3 == 'Advanced':
            st.markdown('**:red[Price included wheeling charge*]**')
            slider_p3 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, 4.62, key='slider_p3', step=0.01, on_change=update_numin)
            num_p3 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, 4.62, key='num_p3', step=0.01, on_change=update_slider)
            st.caption("N-1 trade period price is 4.62 ฿/kWh")
        else:
            st.markdown('**:red[Price included wheeling charge*]**')
            st.markdown(':orange[Using Price Bid = 4.62 ฿/kWh]')
            st.caption("N-1 trade period price is 4.62 ฿/kWh")
            slider_p3 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, value = 4.62, key='slider_p3', step=0.01, on_change=update_numin1, disabled=True) 
            num_p3 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, value = 4.62, key='num_p3', step=0.01, on_change=update_slider1, disabled=True)
            num_p3 = 4.62
        q3 = num_q3
        p3 = num_p3
        st.markdown("---")
        st.markdown("Bid Details for Player 3:")
        st.markdown("Quantity = " + ":red[" + "{:.2f}".format(q3) + "] " + "kWh")
        st.markdown("Price* = " + ":red[" + "{:.2f}".format(p3) + "] " + "฿/kWh")
        st.caption(f':red_circle: Price for matching will be {p3-0.87:.2f} ฿/kWh')
        st.caption('*Included wheeling charge 0.87 THB (฿)')
    with col4:
        st.markdown(":four: **Player 4 :blue[- Seller: ]**")
        mode4 = st.radio("Select Operation Mode", ["Automatic", "Advanced"],horizontal=True,key='mode4', index=1)
        st.markdown('**:blue[Quantity from actual]**')
        slide_q4 = st.slider('Select Quantity Bid (kWh):', 0, 30, key='slider_q4', step=1, on_change=update_numin)
        num_q4 = st.number_input('Enter Quantity Bid (kWh):',0, 30, key='num_q4', step=1, on_change=update_slider)
        if mode4 == 'Advanced':
            st.markdown('**:blue[Price]**')
            slider_p4 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, 3.75, key='slider_p4', step=0.01, on_change=update_numin)
            num_p4 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, 3.75, key='num_p4', step=0.01, on_change=update_slider)
            st.caption("N-1 trade period price is 3.75 ฿/kWh")
        else:
            st.markdown('**:blue[Price]**')
            st.markdown(':orange[Using Price Bid = 3.75 ฿/kWh]')
            st.caption("N-1 trade period price is 3.75 ฿/kWh")
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
        slide_q5 = st.slider('Select Quantity Bid (kWh):', 0, 30, key='slider_q5', step=1, on_change=update_numin)
        num_q5 = st.number_input('Enter Quantity Bid (kWh):',0, 30, key='num_q5', step=1, on_change=update_slider)
        if mode5 == 'Advanced':
            st.markdown('**:blue[Price]**')
            slider_p5 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, 3.75, key='slider_p5', step=0.01, on_change=update_numin)
            num_p5 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, 3.75, key='num_p5', step=0.01, on_change=update_slider)
            st.caption("N-1 trade period price is 3.75 ฿/kWh")
        else:
            st.markdown('**:blue[Price]**')
            st.markdown(':orange[Using Price Bid = 3.75 ฿/kWh]')
            st.caption("N-1 trade period price is 3.75 ฿/kWh")
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
        slide_q6 = st.slider('Select Quantity Bid (kWh):', 0, 30, key='slider_q6', step=1, on_change=update_numin)
        num_q6 = st.number_input('Enter Quantity Bid (kWh):',0, 30, key='num_q6', step=1, on_change=update_slider)
        if mode6 == 'Advanced':
            st.markdown('**:blue[Price]**')
            slider_p6 = st.slider('Select Price Bid* (฿/kWh):', 0.0, 10.0, 3.75, key='slider_p6', step=0.01, on_change=update_numin)
            num_p6 = st.number_input('Enter Price Bid* (฿/kWh):',0.0, 10.0, 3.75, key='num_p6', step=0.01, on_change=update_slider)
            st.caption("N-1 trade period price is 3.75 ฿/kWh")
        else:
            st.markdown('**:blue[Price]**')
            st.markdown(':orange[Using Price Bid = 3.75 ฿/kWh]')
            st.caption("N-1 trade period price is 3.75 ฿/kWh")
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

if st.button(":bar_chart: Run P2P Market Clearing"):
    st.success('Optimization completed.', icon="✅")
    colMatching1, colMatching2, colMatching3, res4 = st.columns([1.5,1.5,5,2])
    with colMatching1:
        sum_q_bid = sum([q1,q2,q3])
        avg_bid = sum([p1,p2,p3])/3
        st.markdown(':red[Demand bids for matching:]')
        st.caption(f'Sum bid quantity = {sum_q_bid}')
        st.caption(f'Average bid quantity = {sum_q_bid/3:.2f}')
        st.caption(f'Average bid price = {avg_bid:.2f}')
        #st.metric(label="Player 1:", value=f"{p1} ฿", delta=f"{(p1-avg_bid):.2f} ฿")
        st.code(f'''Player 1: 
        Quantity = {q1}
        Price* = {p1-0.87:.2f}''')
        st.code(f'''Player 2: 
        Quantity = {q2}
        Price* = {p2-0.87:.2f}''')
        st.code(f'''Player 3: 
        Quantity = {q3}
        Price* = {p3-0.87:.2f}''')
        st.caption('Before matching:')
        st.caption('Excluded wheeling charge 0.87 ฿/kWh')
    with colMatching2:
        sum_q_offer = sum([q4,q5,q6])
        avg_offer = sum([p4,p5,p6])/3
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
        mar = pm.Market()
        mar.accept_bid(q1, round((p1 - 0.87),2), 0, True, 0, True)
        mar.accept_bid(q2, round((p2- 0.87),2), 1, True, 0, True)
        mar.accept_bid(q3, round((p3 - 0.87),2), 2, True, 0, True)
        mar.accept_bid(q4, round(p4,2), 3, False, 0, True)
        mar.accept_bid(q5, round(p5,2), 4, False, 0, True)
        mar.accept_bid(q6, round(p6,2), 5, False, 0, True)
        bids = mar.bm.get_df()
        transactions, extras = mar.run('p2p') # run the p2p mechanism
        #stats = mar.statistics()
        #ax = mar.plot_method('p2p')
        #st.pyplot(ax)
        #st.pyplot(mar.plot_method('p2p'))
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
            Price* = {p1-0.87:.2f}''')
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
                        st.metric("Matched price* with profit","{:.2f} ฿/kWh".format(row['price']),"{:.2f} ฿/kWh discount".format(row['price']-(p1-0.87)),"inverse","Matched price = Pbid + Psharing_profit")
                        st.caption(f"{row['price']+0.87:.2f} ฿/kWh including wheeling charge")
                        st.markdown(f":white_check_mark: Matched with Player {row['userB']}")
                        st.markdown(f"Pay :violet[{(row['price']*row['quantity']):.2f} ฿] to Player {row['userB']}")
                        st.markdown(f"Pay :violet[{0.87*row['quantity']:.2f} ฿] to TSO/DSO")
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
            Price* = {p2-0.87:.2f}''')
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
                        st.metric("Matched price* with profit","{:.2f} ฿/kWh".format(row['price']),"{:.2f} ฿/kWh discount".format(row['price']-(p2-0.87)),"inverse","Matched price = Pbid + Psharing_profit")
                        st.caption(f"{row['price']+0.87:.2f} ฿/kWh including wheeling charge")
                        st.markdown(f":white_check_mark: Matched with Player {row['userB']}")
                        st.markdown(f"Pay :violet[{(row['price']*row['quantity']):.2f} ฿] to Player {row['userB']}")
                        st.markdown(f"Pay :violet[{0.87*row['quantity']:.2f} ฿] to TSO/DSO")
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
            Price* = {p3-0.87:.2f}''')
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
                        st.metric("Matched price* with profit","{:.2f} ฿/kWh".format(row['price']),"{:.2f} ฿/kWh discount".format(row['price']-(p3-0.87)),"inverse","Matched price = Pbid + Psharing_profit")
                        st.caption(f"{row['price']+0.87:.2f} ฿/kWh including wheeling charge")
                        st.markdown(f":white_check_mark: Matched with Player {row['userB']}")
                        st.markdown(f"Pay :violet[{(row['price']*row['quantity']):.2f} ฿] to Player {row['userB']}")
                        st.markdown(f"Pay :violet[{0.87*row['quantity']:.2f} ฿] to TSO/DSO")
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
                        st.caption('<font color="grey">–</font>', unsafe_allow_html=True)
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
                        st.caption('<font color="grey">–</font>', unsafe_allow_html=True)
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
                        st.caption('<font color="grey">–</font>', unsafe_allow_html=True)
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
