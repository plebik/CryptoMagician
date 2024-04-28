import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_rankings(ranking_data: dict):
    fig_ = make_subplots(rows=len(ranking_data), cols=1, subplot_titles=list(ranking_data.keys()),
                         horizontal_spacing=0.8, row_heights=[0.9] * len(ranking_data))
    for i, (key, df) in enumerate(ranking_data.items(), start=1):
        # Add a subplot for each DataFrame
        for n in df.columns:
            fig_.add_trace(
                go.Bar(name=n, x=df.index, y=df[n], legendgroup=key),
                row=i,
                col=1
            )

    fig_.update_layout(
        height=1550,
        width=1000,
        title="Ranking divied to intervals",
        barmode='group',
        legend_tracegroupgap=112
    )

    fig_.update_xaxes(title_text='Place in ranking')
    fig_.update_yaxes(title_text='Counts')

    return fig_
