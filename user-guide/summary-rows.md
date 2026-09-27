# Summary Rows

Summary rows provide aggregated values (such as totals, means, or counts) directly in the table, adjacent to the data they summarize. **Great Tables** supports two types: group-level summaries that appear next to each row group, and grand summaries that aggregate across the entire table. Both types let you define multiple aggregation functions at once and control where the summary appears.

Embedding aggregations directly in the table keeps the reader's eye on the data. When a total or average lives in a separate paragraph or a footnote, the reader has to scan back and forth between the table and the surrounding text to compare a group's summary to its individual rows. Summary rows eliminate that friction by placing the computed values right next to the data they describe.


# Setting Up the Example Data

For these examples, we will use a sales dataset with row groups representing different product categories.


``` python
import polars as pl
from great_tables import GT

sales_df = pl.DataFrame({
    "product": ["Laptop", "Mouse", "Keyboard", "Monitor", "Webcam", "Headset"],
    "category": ["Computing", "Computing", "Computing", "Peripherals", "Peripherals", "Peripherals"],
    "units_sold": [45, 230, 180, 65, 120, 95],
    "revenue": [67500, 4600, 9000, 19500, 6000, 7125],
})

gt_sales = (
    GT(sales_df, rowname_col="product", groupname_col="category")
    .tab_header(title="Q4 Product Sales", subtitle="By category")
    .fmt_number(columns="revenue", decimals=0, use_seps=True)
)

gt_sales
```


<style>
#qlqquilqgc table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#qlqquilqgc thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#qlqquilqgc p { margin: 0; padding: 0; }
 #qlqquilqgc .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #qlqquilqgc .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #qlqquilqgc .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #qlqquilqgc .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #qlqquilqgc .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #qlqquilqgc .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #qlqquilqgc .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #qlqquilqgc .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #qlqquilqgc .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #qlqquilqgc .gt_column_spanner_outer:first-child { padding-left: 0; }
 #qlqquilqgc .gt_column_spanner_outer:last-child { padding-right: 0; }
 #qlqquilqgc .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #qlqquilqgc .gt_spanner_row { border-bottom-style: hidden; }
 #qlqquilqgc .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #qlqquilqgc .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #qlqquilqgc .gt_from_md> :first-child { margin-top: 0; }
 #qlqquilqgc .gt_from_md> :last-child { margin-bottom: 0; }
 #qlqquilqgc .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #qlqquilqgc .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #qlqquilqgc .gt_indent_1 { text-indent: 5px; }
 #qlqquilqgc .gt_indent_2 { text-indent: calc(5px * 2); }
 #qlqquilqgc .gt_indent_3 { text-indent: calc(5px * 3); }
 #qlqquilqgc .gt_indent_4 { text-indent: calc(5px * 4); }
 #qlqquilqgc .gt_indent_5 { text-indent: calc(5px * 5); }
 #qlqquilqgc .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #qlqquilqgc .gt_row_group_first td { border-top-width: 2px; }
 #qlqquilqgc .gt_row_group_first th { border-top-width: 2px; }
 #qlqquilqgc .gt_striped { color: #333333; background-color: #F4F4F4; }
 #qlqquilqgc .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #qlqquilqgc .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #qlqquilqgc .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #qlqquilqgc .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #qlqquilqgc .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #qlqquilqgc .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #qlqquilqgc .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #qlqquilqgc .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #qlqquilqgc .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #qlqquilqgc .gt_left { text-align: left; }
 #qlqquilqgc .gt_center { text-align: center; }
 #qlqquilqgc .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #qlqquilqgc .gt_font_normal { font-weight: normal; }
 #qlqquilqgc .gt_font_bold { font-weight: bold; }
 #qlqquilqgc .gt_font_italic { font-style: italic; }
 #qlqquilqgc .gt_super { font-size: 65%; }
 #qlqquilqgc .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #qlqquilqgc .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #qlqquilqgc .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #qlqquilqgc .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #qlqquilqgc .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #qlqquilqgc .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_title gt_font_normal">Q4 Product Sales</th>
</tr>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">By category</th>
</tr>
<tr class="gt_col_headings">
<th class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="units_sold" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">units_sold</th>
<th id="revenue" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">revenue</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="3" class="gt_group_heading">Computing</th>
</tr>

<tr>
<td class="gt_row gt_left gt_stub">Laptop</td>
<td class="gt_row gt_right">45</td>
<td class="gt_row gt_right">67,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Mouse</td>
<td class="gt_row gt_right">230</td>
<td class="gt_row gt_right">4,600</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Keyboard</td>
<td class="gt_row gt_right">180</td>
<td class="gt_row gt_right">9,000</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="3" class="gt_group_heading">Peripherals</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Monitor</td>
<td class="gt_row gt_right">65</td>
<td class="gt_row gt_right">19,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Webcam</td>
<td class="gt_row gt_right">120</td>
<td class="gt_row gt_right">6,000</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Headset</td>
<td class="gt_row gt_right">95</td>
<td class="gt_row gt_right">7,125</td>
</tr>
</tbody>
</table>


This table has two row groups: `"Computing"` and `"Peripherals"`. We can now add summaries at the group level and at the grand level.


# Group-Level Summary Rows

The [summary_rows()](../reference/GT.summary_rows.md#great_tables.GT.summary_rows) method adds summary rows to each row group. You provide aggregation functions through the `fns=` argument as a dictionary, where keys become the summary row labels and values are the aggregation logic.

When using a **Polars** DataFrame, the aggregation values should be Polars expressions.


``` python
(
    gt_sales
    .summary_rows(
        fns={"Total": pl.col("units_sold", "revenue").sum()}
    )
)
```


<style>
#uenxmjxnlj table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#uenxmjxnlj thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#uenxmjxnlj p { margin: 0; padding: 0; }
 #uenxmjxnlj .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #uenxmjxnlj .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #uenxmjxnlj .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #uenxmjxnlj .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #uenxmjxnlj .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #uenxmjxnlj .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #uenxmjxnlj .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #uenxmjxnlj .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #uenxmjxnlj .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #uenxmjxnlj .gt_column_spanner_outer:first-child { padding-left: 0; }
 #uenxmjxnlj .gt_column_spanner_outer:last-child { padding-right: 0; }
 #uenxmjxnlj .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #uenxmjxnlj .gt_spanner_row { border-bottom-style: hidden; }
 #uenxmjxnlj .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #uenxmjxnlj .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #uenxmjxnlj .gt_from_md> :first-child { margin-top: 0; }
 #uenxmjxnlj .gt_from_md> :last-child { margin-bottom: 0; }
 #uenxmjxnlj .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #uenxmjxnlj .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #uenxmjxnlj .gt_indent_1 { text-indent: 5px; }
 #uenxmjxnlj .gt_indent_2 { text-indent: calc(5px * 2); }
 #uenxmjxnlj .gt_indent_3 { text-indent: calc(5px * 3); }
 #uenxmjxnlj .gt_indent_4 { text-indent: calc(5px * 4); }
 #uenxmjxnlj .gt_indent_5 { text-indent: calc(5px * 5); }
 #uenxmjxnlj .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #uenxmjxnlj .gt_row_group_first td { border-top-width: 2px; }
 #uenxmjxnlj .gt_row_group_first th { border-top-width: 2px; }
 #uenxmjxnlj .gt_striped { color: #333333; background-color: #F4F4F4; }
 #uenxmjxnlj .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #uenxmjxnlj .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #uenxmjxnlj .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #uenxmjxnlj .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #uenxmjxnlj .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #uenxmjxnlj .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #uenxmjxnlj .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #uenxmjxnlj .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #uenxmjxnlj .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #uenxmjxnlj .gt_left { text-align: left; }
 #uenxmjxnlj .gt_center { text-align: center; }
 #uenxmjxnlj .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #uenxmjxnlj .gt_font_normal { font-weight: normal; }
 #uenxmjxnlj .gt_font_bold { font-weight: bold; }
 #uenxmjxnlj .gt_font_italic { font-style: italic; }
 #uenxmjxnlj .gt_super { font-size: 65%; }
 #uenxmjxnlj .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #uenxmjxnlj .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #uenxmjxnlj .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #uenxmjxnlj .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #uenxmjxnlj .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #uenxmjxnlj .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_title gt_font_normal">Q4 Product Sales</th>
</tr>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">By category</th>
</tr>
<tr class="gt_col_headings">
<th class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="units_sold" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">units_sold</th>
<th id="revenue" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">revenue</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="3" class="gt_group_heading">Computing</th>
</tr>

<tr>
<td class="gt_row gt_left gt_stub">Laptop</td>
<td class="gt_row gt_right">45</td>
<td class="gt_row gt_right">67,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Mouse</td>
<td class="gt_row gt_right">230</td>
<td class="gt_row gt_right">4,600</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Keyboard</td>
<td class="gt_row gt_right">180</td>
<td class="gt_row gt_right">9,000</td>
</tr>
<tr>
<td class="gt_first_summary_row gt_row gt_left gt_stub gt_summary_row">Total</td>
<td class="gt_first_summary_row gt_row gt_right gt_summary_row">455</td>
<td class="gt_first_summary_row gt_row gt_right gt_summary_row">81100</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="3" class="gt_group_heading">Peripherals</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Monitor</td>
<td class="gt_row gt_right">65</td>
<td class="gt_row gt_right">19,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Webcam</td>
<td class="gt_row gt_right">120</td>
<td class="gt_row gt_right">6,000</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Headset</td>
<td class="gt_row gt_right">95</td>
<td class="gt_row gt_right">7,125</td>
</tr>
<tr>
<td class="gt_first_summary_row gt_row gt_left gt_stub gt_summary_row">Total</td>
<td class="gt_first_summary_row gt_row gt_right gt_summary_row">280</td>
<td class="gt_first_summary_row gt_row gt_right gt_summary_row">32625</td>
</tr>
</tbody>
</table>


Each row group now has a `"Total"` summary row at the bottom showing the sum of numeric columns within that group.


## Multiple Aggregation Functions

You can include several functions in the `fns=` dictionary to produce multiple summary rows per group.


``` python
(
    gt_sales
    .summary_rows(
        fns={
            "Total": pl.col("units_sold", "revenue").sum(),
            "Average": pl.col("units_sold", "revenue").mean(),
        }
    )
)
```


<style>
#fhtrsivjsv table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#fhtrsivjsv thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#fhtrsivjsv p { margin: 0; padding: 0; }
 #fhtrsivjsv .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #fhtrsivjsv .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #fhtrsivjsv .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #fhtrsivjsv .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #fhtrsivjsv .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #fhtrsivjsv .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #fhtrsivjsv .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #fhtrsivjsv .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #fhtrsivjsv .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #fhtrsivjsv .gt_column_spanner_outer:first-child { padding-left: 0; }
 #fhtrsivjsv .gt_column_spanner_outer:last-child { padding-right: 0; }
 #fhtrsivjsv .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #fhtrsivjsv .gt_spanner_row { border-bottom-style: hidden; }
 #fhtrsivjsv .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #fhtrsivjsv .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #fhtrsivjsv .gt_from_md> :first-child { margin-top: 0; }
 #fhtrsivjsv .gt_from_md> :last-child { margin-bottom: 0; }
 #fhtrsivjsv .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #fhtrsivjsv .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #fhtrsivjsv .gt_indent_1 { text-indent: 5px; }
 #fhtrsivjsv .gt_indent_2 { text-indent: calc(5px * 2); }
 #fhtrsivjsv .gt_indent_3 { text-indent: calc(5px * 3); }
 #fhtrsivjsv .gt_indent_4 { text-indent: calc(5px * 4); }
 #fhtrsivjsv .gt_indent_5 { text-indent: calc(5px * 5); }
 #fhtrsivjsv .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #fhtrsivjsv .gt_row_group_first td { border-top-width: 2px; }
 #fhtrsivjsv .gt_row_group_first th { border-top-width: 2px; }
 #fhtrsivjsv .gt_striped { color: #333333; background-color: #F4F4F4; }
 #fhtrsivjsv .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #fhtrsivjsv .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #fhtrsivjsv .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #fhtrsivjsv .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #fhtrsivjsv .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #fhtrsivjsv .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #fhtrsivjsv .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #fhtrsivjsv .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #fhtrsivjsv .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #fhtrsivjsv .gt_left { text-align: left; }
 #fhtrsivjsv .gt_center { text-align: center; }
 #fhtrsivjsv .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #fhtrsivjsv .gt_font_normal { font-weight: normal; }
 #fhtrsivjsv .gt_font_bold { font-weight: bold; }
 #fhtrsivjsv .gt_font_italic { font-style: italic; }
 #fhtrsivjsv .gt_super { font-size: 65%; }
 #fhtrsivjsv .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #fhtrsivjsv .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #fhtrsivjsv .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #fhtrsivjsv .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #fhtrsivjsv .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #fhtrsivjsv .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_title gt_font_normal">Q4 Product Sales</th>
</tr>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">By category</th>
</tr>
<tr class="gt_col_headings">
<th class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="units_sold" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">units_sold</th>
<th id="revenue" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">revenue</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="3" class="gt_group_heading">Computing</th>
</tr>

<tr>
<td class="gt_row gt_left gt_stub">Laptop</td>
<td class="gt_row gt_right">45</td>
<td class="gt_row gt_right">67,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Mouse</td>
<td class="gt_row gt_right">230</td>
<td class="gt_row gt_right">4,600</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Keyboard</td>
<td class="gt_row gt_right">180</td>
<td class="gt_row gt_right">9,000</td>
</tr>
<tr>
<td class="gt_first_summary_row gt_row gt_left gt_stub gt_summary_row">Total</td>
<td class="gt_first_summary_row gt_row gt_right gt_summary_row">455</td>
<td class="gt_first_summary_row gt_row gt_right gt_summary_row">81100</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub gt_summary_row">Average</td>
<td class="gt_row gt_right gt_summary_row">151.66666666666666</td>
<td class="gt_row gt_right gt_summary_row">27033.333333333332</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="3" class="gt_group_heading">Peripherals</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Monitor</td>
<td class="gt_row gt_right">65</td>
<td class="gt_row gt_right">19,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Webcam</td>
<td class="gt_row gt_right">120</td>
<td class="gt_row gt_right">6,000</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Headset</td>
<td class="gt_row gt_right">95</td>
<td class="gt_row gt_right">7,125</td>
</tr>
<tr>
<td class="gt_first_summary_row gt_row gt_left gt_stub gt_summary_row">Total</td>
<td class="gt_first_summary_row gt_row gt_right gt_summary_row">280</td>
<td class="gt_first_summary_row gt_row gt_right gt_summary_row">32625</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub gt_summary_row">Average</td>
<td class="gt_row gt_right gt_summary_row">93.33333333333333</td>
<td class="gt_row gt_right gt_summary_row">10875.0</td>
</tr>
</tbody>
</table>


Both a `"Total"` and an `"Average"` row now appear at the bottom of each group.


## Placing Summaries at the Top

By default, summary rows appear at the bottom of each group. You can place them at the top instead by setting `side="top"`.


``` python
(
    gt_sales
    .summary_rows(
        fns={"Total": pl.col("units_sold", "revenue").sum()},
        side="top"
    )
)
```


<style>
#csqrlyvwyq table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#csqrlyvwyq thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#csqrlyvwyq p { margin: 0; padding: 0; }
 #csqrlyvwyq .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #csqrlyvwyq .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #csqrlyvwyq .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #csqrlyvwyq .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #csqrlyvwyq .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #csqrlyvwyq .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #csqrlyvwyq .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #csqrlyvwyq .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #csqrlyvwyq .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #csqrlyvwyq .gt_column_spanner_outer:first-child { padding-left: 0; }
 #csqrlyvwyq .gt_column_spanner_outer:last-child { padding-right: 0; }
 #csqrlyvwyq .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #csqrlyvwyq .gt_spanner_row { border-bottom-style: hidden; }
 #csqrlyvwyq .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #csqrlyvwyq .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #csqrlyvwyq .gt_from_md> :first-child { margin-top: 0; }
 #csqrlyvwyq .gt_from_md> :last-child { margin-bottom: 0; }
 #csqrlyvwyq .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #csqrlyvwyq .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #csqrlyvwyq .gt_indent_1 { text-indent: 5px; }
 #csqrlyvwyq .gt_indent_2 { text-indent: calc(5px * 2); }
 #csqrlyvwyq .gt_indent_3 { text-indent: calc(5px * 3); }
 #csqrlyvwyq .gt_indent_4 { text-indent: calc(5px * 4); }
 #csqrlyvwyq .gt_indent_5 { text-indent: calc(5px * 5); }
 #csqrlyvwyq .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #csqrlyvwyq .gt_row_group_first td { border-top-width: 2px; }
 #csqrlyvwyq .gt_row_group_first th { border-top-width: 2px; }
 #csqrlyvwyq .gt_striped { color: #333333; background-color: #F4F4F4; }
 #csqrlyvwyq .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #csqrlyvwyq .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #csqrlyvwyq .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #csqrlyvwyq .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #csqrlyvwyq .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #csqrlyvwyq .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #csqrlyvwyq .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #csqrlyvwyq .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #csqrlyvwyq .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #csqrlyvwyq .gt_left { text-align: left; }
 #csqrlyvwyq .gt_center { text-align: center; }
 #csqrlyvwyq .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #csqrlyvwyq .gt_font_normal { font-weight: normal; }
 #csqrlyvwyq .gt_font_bold { font-weight: bold; }
 #csqrlyvwyq .gt_font_italic { font-style: italic; }
 #csqrlyvwyq .gt_super { font-size: 65%; }
 #csqrlyvwyq .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #csqrlyvwyq .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #csqrlyvwyq .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #csqrlyvwyq .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #csqrlyvwyq .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #csqrlyvwyq .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_title gt_font_normal">Q4 Product Sales</th>
</tr>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">By category</th>
</tr>
<tr class="gt_col_headings">
<th class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="units_sold" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">units_sold</th>
<th id="revenue" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">revenue</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="3" class="gt_group_heading">Computing</th>
</tr>

<tr>
<td class="gt_last_summary_row_top gt_row gt_left gt_stub gt_summary_row">Total</td>
<td class="gt_last_summary_row_top gt_row gt_right gt_summary_row">455</td>
<td class="gt_last_summary_row_top gt_row gt_right gt_summary_row">81100</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Laptop</td>
<td class="gt_row gt_right">45</td>
<td class="gt_row gt_right">67,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Mouse</td>
<td class="gt_row gt_right">230</td>
<td class="gt_row gt_right">4,600</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Keyboard</td>
<td class="gt_row gt_right">180</td>
<td class="gt_row gt_right">9,000</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="3" class="gt_group_heading">Peripherals</td>
</tr>
<tr>
<td class="gt_last_summary_row_top gt_row gt_left gt_stub gt_summary_row">Total</td>
<td class="gt_last_summary_row_top gt_row gt_right gt_summary_row">280</td>
<td class="gt_last_summary_row_top gt_row gt_right gt_summary_row">32625</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Monitor</td>
<td class="gt_row gt_right">65</td>
<td class="gt_row gt_right">19,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Webcam</td>
<td class="gt_row gt_right">120</td>
<td class="gt_row gt_right">6,000</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Headset</td>
<td class="gt_row gt_right">95</td>
<td class="gt_row gt_right">7,125</td>
</tr>
</tbody>
</table>


The summary row now sits above the data rows in each group rather than below them, making the totals immediately visible.

Top placement works well when the summary is the main takeaway and the detail rows serve as supporting evidence. Readers see the headline number first and can drill into the breakdown below it. Bottom placement (the default) follows the more conventional reading order where you examine the data first and arrive at the summary as a natural conclusion.


## Targeting Specific Groups

If you only want summaries for certain groups, use the `groups=` argument with a list of group names.


``` python
(
    gt_sales
    .summary_rows(
        fns={"Total": pl.col("units_sold", "revenue").sum()},
        groups=["Computing"]
    )
)
```


<style>
#ynrmzxhvjb table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#ynrmzxhvjb thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#ynrmzxhvjb p { margin: 0; padding: 0; }
 #ynrmzxhvjb .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #ynrmzxhvjb .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #ynrmzxhvjb .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #ynrmzxhvjb .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #ynrmzxhvjb .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #ynrmzxhvjb .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ynrmzxhvjb .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #ynrmzxhvjb .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #ynrmzxhvjb .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #ynrmzxhvjb .gt_column_spanner_outer:first-child { padding-left: 0; }
 #ynrmzxhvjb .gt_column_spanner_outer:last-child { padding-right: 0; }
 #ynrmzxhvjb .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #ynrmzxhvjb .gt_spanner_row { border-bottom-style: hidden; }
 #ynrmzxhvjb .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #ynrmzxhvjb .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #ynrmzxhvjb .gt_from_md> :first-child { margin-top: 0; }
 #ynrmzxhvjb .gt_from_md> :last-child { margin-bottom: 0; }
 #ynrmzxhvjb .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #ynrmzxhvjb .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #ynrmzxhvjb .gt_indent_1 { text-indent: 5px; }
 #ynrmzxhvjb .gt_indent_2 { text-indent: calc(5px * 2); }
 #ynrmzxhvjb .gt_indent_3 { text-indent: calc(5px * 3); }
 #ynrmzxhvjb .gt_indent_4 { text-indent: calc(5px * 4); }
 #ynrmzxhvjb .gt_indent_5 { text-indent: calc(5px * 5); }
 #ynrmzxhvjb .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #ynrmzxhvjb .gt_row_group_first td { border-top-width: 2px; }
 #ynrmzxhvjb .gt_row_group_first th { border-top-width: 2px; }
 #ynrmzxhvjb .gt_striped { color: #333333; background-color: #F4F4F4; }
 #ynrmzxhvjb .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ynrmzxhvjb .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #ynrmzxhvjb .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #ynrmzxhvjb .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ynrmzxhvjb .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #ynrmzxhvjb .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #ynrmzxhvjb .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #ynrmzxhvjb .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ynrmzxhvjb .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #ynrmzxhvjb .gt_left { text-align: left; }
 #ynrmzxhvjb .gt_center { text-align: center; }
 #ynrmzxhvjb .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #ynrmzxhvjb .gt_font_normal { font-weight: normal; }
 #ynrmzxhvjb .gt_font_bold { font-weight: bold; }
 #ynrmzxhvjb .gt_font_italic { font-style: italic; }
 #ynrmzxhvjb .gt_super { font-size: 65%; }
 #ynrmzxhvjb .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ynrmzxhvjb .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #ynrmzxhvjb .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ynrmzxhvjb .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #ynrmzxhvjb .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #ynrmzxhvjb .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_title gt_font_normal">Q4 Product Sales</th>
</tr>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">By category</th>
</tr>
<tr class="gt_col_headings">
<th class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="units_sold" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">units_sold</th>
<th id="revenue" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">revenue</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="3" class="gt_group_heading">Computing</th>
</tr>

<tr>
<td class="gt_row gt_left gt_stub">Laptop</td>
<td class="gt_row gt_right">45</td>
<td class="gt_row gt_right">67,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Mouse</td>
<td class="gt_row gt_right">230</td>
<td class="gt_row gt_right">4,600</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Keyboard</td>
<td class="gt_row gt_right">180</td>
<td class="gt_row gt_right">9,000</td>
</tr>
<tr>
<td class="gt_first_summary_row gt_row gt_left gt_stub gt_summary_row">Total</td>
<td class="gt_first_summary_row gt_row gt_right gt_summary_row">455</td>
<td class="gt_first_summary_row gt_row gt_right gt_summary_row">81100</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="3" class="gt_group_heading">Peripherals</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Monitor</td>
<td class="gt_row gt_right">65</td>
<td class="gt_row gt_right">19,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Webcam</td>
<td class="gt_row gt_right">120</td>
<td class="gt_row gt_right">6,000</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Headset</td>
<td class="gt_row gt_right">95</td>
<td class="gt_row gt_right">7,125</td>
</tr>
</tbody>
</table>


Only the `"Computing"` group receives a summary row.


# Grand Summary Rows

The [grand_summary_rows()](../reference/GT.grand_summary_rows.md#great_tables.GT.grand_summary_rows) method works the same way as [summary_rows()](../reference/GT.summary_rows.md#great_tables.GT.summary_rows), but it aggregates across all data in the table regardless of row groups. The resulting summary rows appear at the very bottom (or top) of the table.


``` python
(
    gt_sales
    .grand_summary_rows(
        fns={"Grand Total": pl.col("units_sold", "revenue").sum()}
    )
)
```


<style>
#prmqaxjbqb table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#prmqaxjbqb thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#prmqaxjbqb p { margin: 0; padding: 0; }
 #prmqaxjbqb .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #prmqaxjbqb .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #prmqaxjbqb .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #prmqaxjbqb .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #prmqaxjbqb .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #prmqaxjbqb .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #prmqaxjbqb .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #prmqaxjbqb .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #prmqaxjbqb .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #prmqaxjbqb .gt_column_spanner_outer:first-child { padding-left: 0; }
 #prmqaxjbqb .gt_column_spanner_outer:last-child { padding-right: 0; }
 #prmqaxjbqb .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #prmqaxjbqb .gt_spanner_row { border-bottom-style: hidden; }
 #prmqaxjbqb .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #prmqaxjbqb .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #prmqaxjbqb .gt_from_md> :first-child { margin-top: 0; }
 #prmqaxjbqb .gt_from_md> :last-child { margin-bottom: 0; }
 #prmqaxjbqb .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #prmqaxjbqb .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #prmqaxjbqb .gt_indent_1 { text-indent: 5px; }
 #prmqaxjbqb .gt_indent_2 { text-indent: calc(5px * 2); }
 #prmqaxjbqb .gt_indent_3 { text-indent: calc(5px * 3); }
 #prmqaxjbqb .gt_indent_4 { text-indent: calc(5px * 4); }
 #prmqaxjbqb .gt_indent_5 { text-indent: calc(5px * 5); }
 #prmqaxjbqb .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #prmqaxjbqb .gt_row_group_first td { border-top-width: 2px; }
 #prmqaxjbqb .gt_row_group_first th { border-top-width: 2px; }
 #prmqaxjbqb .gt_striped { color: #333333; background-color: #F4F4F4; }
 #prmqaxjbqb .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #prmqaxjbqb .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #prmqaxjbqb .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #prmqaxjbqb .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #prmqaxjbqb .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #prmqaxjbqb .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #prmqaxjbqb .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #prmqaxjbqb .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #prmqaxjbqb .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #prmqaxjbqb .gt_left { text-align: left; }
 #prmqaxjbqb .gt_center { text-align: center; }
 #prmqaxjbqb .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #prmqaxjbqb .gt_font_normal { font-weight: normal; }
 #prmqaxjbqb .gt_font_bold { font-weight: bold; }
 #prmqaxjbqb .gt_font_italic { font-style: italic; }
 #prmqaxjbqb .gt_super { font-size: 65%; }
 #prmqaxjbqb .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #prmqaxjbqb .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #prmqaxjbqb .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #prmqaxjbqb .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #prmqaxjbqb .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #prmqaxjbqb .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_title gt_font_normal">Q4 Product Sales</th>
</tr>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">By category</th>
</tr>
<tr class="gt_col_headings">
<th class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="units_sold" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">units_sold</th>
<th id="revenue" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">revenue</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="3" class="gt_group_heading">Computing</th>
</tr>

<tr>
<td class="gt_row gt_left gt_stub">Laptop</td>
<td class="gt_row gt_right">45</td>
<td class="gt_row gt_right">67,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Mouse</td>
<td class="gt_row gt_right">230</td>
<td class="gt_row gt_right">4,600</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Keyboard</td>
<td class="gt_row gt_right">180</td>
<td class="gt_row gt_right">9,000</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="3" class="gt_group_heading">Peripherals</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Monitor</td>
<td class="gt_row gt_right">65</td>
<td class="gt_row gt_right">19,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Webcam</td>
<td class="gt_row gt_right">120</td>
<td class="gt_row gt_right">6,000</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Headset</td>
<td class="gt_row gt_right">95</td>
<td class="gt_row gt_right">7,125</td>
</tr>
<tr>
<td class="gt_first_grand_summary_row_bottom gt_row gt_left gt_stub gt_grand_summary_row">Grand Total</td>
<td class="gt_first_grand_summary_row_bottom gt_row gt_right gt_grand_summary_row">735</td>
<td class="gt_first_grand_summary_row_bottom gt_row gt_right gt_grand_summary_row">113725</td>
</tr>
</tbody>
</table>


A single `"Grand Total"` row appears below all row groups, showing the overall totals.


## Combining Group and Grand Summaries

You can use both [summary_rows()](../reference/GT.summary_rows.md#great_tables.GT.summary_rows) and [grand_summary_rows()](../reference/GT.grand_summary_rows.md#great_tables.GT.grand_summary_rows) on the same table to provide aggregation at both levels.


``` python
(
    gt_sales
    .summary_rows(
        fns={"Subtotal": pl.col("units_sold", "revenue").sum()}
    )
    .grand_summary_rows(
        fns={
            "Grand Total": pl.col("units_sold", "revenue").sum(),
            "Overall Average": pl.col("units_sold", "revenue").mean(),
        }
    )
)
```


<style>
#nlcvgybmrz table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#nlcvgybmrz thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#nlcvgybmrz p { margin: 0; padding: 0; }
 #nlcvgybmrz .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #nlcvgybmrz .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #nlcvgybmrz .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #nlcvgybmrz .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #nlcvgybmrz .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #nlcvgybmrz .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #nlcvgybmrz .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #nlcvgybmrz .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #nlcvgybmrz .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #nlcvgybmrz .gt_column_spanner_outer:first-child { padding-left: 0; }
 #nlcvgybmrz .gt_column_spanner_outer:last-child { padding-right: 0; }
 #nlcvgybmrz .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #nlcvgybmrz .gt_spanner_row { border-bottom-style: hidden; }
 #nlcvgybmrz .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #nlcvgybmrz .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #nlcvgybmrz .gt_from_md> :first-child { margin-top: 0; }
 #nlcvgybmrz .gt_from_md> :last-child { margin-bottom: 0; }
 #nlcvgybmrz .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #nlcvgybmrz .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #nlcvgybmrz .gt_indent_1 { text-indent: 5px; }
 #nlcvgybmrz .gt_indent_2 { text-indent: calc(5px * 2); }
 #nlcvgybmrz .gt_indent_3 { text-indent: calc(5px * 3); }
 #nlcvgybmrz .gt_indent_4 { text-indent: calc(5px * 4); }
 #nlcvgybmrz .gt_indent_5 { text-indent: calc(5px * 5); }
 #nlcvgybmrz .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #nlcvgybmrz .gt_row_group_first td { border-top-width: 2px; }
 #nlcvgybmrz .gt_row_group_first th { border-top-width: 2px; }
 #nlcvgybmrz .gt_striped { color: #333333; background-color: #F4F4F4; }
 #nlcvgybmrz .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #nlcvgybmrz .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #nlcvgybmrz .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #nlcvgybmrz .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #nlcvgybmrz .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #nlcvgybmrz .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #nlcvgybmrz .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #nlcvgybmrz .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #nlcvgybmrz .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #nlcvgybmrz .gt_left { text-align: left; }
 #nlcvgybmrz .gt_center { text-align: center; }
 #nlcvgybmrz .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #nlcvgybmrz .gt_font_normal { font-weight: normal; }
 #nlcvgybmrz .gt_font_bold { font-weight: bold; }
 #nlcvgybmrz .gt_font_italic { font-style: italic; }
 #nlcvgybmrz .gt_super { font-size: 65%; }
 #nlcvgybmrz .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #nlcvgybmrz .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #nlcvgybmrz .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #nlcvgybmrz .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #nlcvgybmrz .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #nlcvgybmrz .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_title gt_font_normal">Q4 Product Sales</th>
</tr>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">By category</th>
</tr>
<tr class="gt_col_headings">
<th class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="units_sold" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">units_sold</th>
<th id="revenue" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">revenue</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="3" class="gt_group_heading">Computing</th>
</tr>

<tr>
<td class="gt_row gt_left gt_stub">Laptop</td>
<td class="gt_row gt_right">45</td>
<td class="gt_row gt_right">67,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Mouse</td>
<td class="gt_row gt_right">230</td>
<td class="gt_row gt_right">4,600</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Keyboard</td>
<td class="gt_row gt_right">180</td>
<td class="gt_row gt_right">9,000</td>
</tr>
<tr>
<td class="gt_first_summary_row gt_row gt_left gt_stub gt_summary_row">Subtotal</td>
<td class="gt_first_summary_row gt_row gt_right gt_summary_row">455</td>
<td class="gt_first_summary_row gt_row gt_right gt_summary_row">81100</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="3" class="gt_group_heading">Peripherals</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Monitor</td>
<td class="gt_row gt_right">65</td>
<td class="gt_row gt_right">19,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Webcam</td>
<td class="gt_row gt_right">120</td>
<td class="gt_row gt_right">6,000</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Headset</td>
<td class="gt_row gt_right">95</td>
<td class="gt_row gt_right">7,125</td>
</tr>
<tr>
<td class="gt_first_summary_row gt_row gt_left gt_stub gt_summary_row">Subtotal</td>
<td class="gt_first_summary_row gt_row gt_right gt_summary_row">280</td>
<td class="gt_first_summary_row gt_row gt_right gt_summary_row">32625</td>
</tr>
<tr>
<td class="gt_first_grand_summary_row_bottom gt_row gt_left gt_stub gt_grand_summary_row">Grand Total</td>
<td class="gt_first_grand_summary_row_bottom gt_row gt_right gt_grand_summary_row">735</td>
<td class="gt_first_grand_summary_row_bottom gt_row gt_right gt_grand_summary_row">113725</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub gt_grand_summary_row">Overall Average</td>
<td class="gt_row gt_right gt_grand_summary_row">122.5</td>
<td class="gt_row gt_right gt_grand_summary_row">18954.166666666668</td>
</tr>
</tbody>
</table>


Each group now has a `"Subtotal"` row, and the table finishes with a `"Grand Total"` and `"Overall Average"` row that span across all groups.

This combination of group and grand summaries is common in financial and sales reporting, where readers expect both subtotals per category and an overall total. The visual hierarchy (group summaries, then a grand summary at the bottom) mirrors how people naturally reason about grouped data: understand the parts first, then see how they add up to the whole.


# Working with Pandas DataFrames

When using a **Pandas** DataFrame, the aggregation functions receive a Pandas DataFrame and should work accordingly.


``` python
import pandas as pd

sales_pd = pd.DataFrame({
    "product": ["Laptop", "Mouse", "Keyboard", "Monitor", "Webcam", "Headset"],
    "category": ["Computing", "Computing", "Computing", "Peripherals", "Peripherals", "Peripherals"],
    "units_sold": [45, 230, 180, 65, 120, 95],
    "revenue": [67500, 4600, 9000, 19500, 6000, 7125],
})

(
    GT(sales_pd, rowname_col="product", groupname_col="category")
    .fmt_number(columns="revenue", decimals=0, use_seps=True)
    .grand_summary_rows(
        fns={"Total": lambda df: df.sum(numeric_only=True)}
    )
)
```


<style>
#esuuttrtvu table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#esuuttrtvu thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#esuuttrtvu p { margin: 0; padding: 0; }
 #esuuttrtvu .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #esuuttrtvu .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #esuuttrtvu .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #esuuttrtvu .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #esuuttrtvu .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #esuuttrtvu .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #esuuttrtvu .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #esuuttrtvu .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #esuuttrtvu .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #esuuttrtvu .gt_column_spanner_outer:first-child { padding-left: 0; }
 #esuuttrtvu .gt_column_spanner_outer:last-child { padding-right: 0; }
 #esuuttrtvu .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #esuuttrtvu .gt_spanner_row { border-bottom-style: hidden; }
 #esuuttrtvu .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #esuuttrtvu .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #esuuttrtvu .gt_from_md> :first-child { margin-top: 0; }
 #esuuttrtvu .gt_from_md> :last-child { margin-bottom: 0; }
 #esuuttrtvu .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #esuuttrtvu .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #esuuttrtvu .gt_indent_1 { text-indent: 5px; }
 #esuuttrtvu .gt_indent_2 { text-indent: calc(5px * 2); }
 #esuuttrtvu .gt_indent_3 { text-indent: calc(5px * 3); }
 #esuuttrtvu .gt_indent_4 { text-indent: calc(5px * 4); }
 #esuuttrtvu .gt_indent_5 { text-indent: calc(5px * 5); }
 #esuuttrtvu .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #esuuttrtvu .gt_row_group_first td { border-top-width: 2px; }
 #esuuttrtvu .gt_row_group_first th { border-top-width: 2px; }
 #esuuttrtvu .gt_striped { color: #333333; background-color: #F4F4F4; }
 #esuuttrtvu .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #esuuttrtvu .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #esuuttrtvu .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #esuuttrtvu .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #esuuttrtvu .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #esuuttrtvu .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #esuuttrtvu .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #esuuttrtvu .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #esuuttrtvu .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #esuuttrtvu .gt_left { text-align: left; }
 #esuuttrtvu .gt_center { text-align: center; }
 #esuuttrtvu .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #esuuttrtvu .gt_font_normal { font-weight: normal; }
 #esuuttrtvu .gt_font_bold { font-weight: bold; }
 #esuuttrtvu .gt_font_italic { font-style: italic; }
 #esuuttrtvu .gt_super { font-size: 65%; }
 #esuuttrtvu .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #esuuttrtvu .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #esuuttrtvu .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #esuuttrtvu .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #esuuttrtvu .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #esuuttrtvu .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="units_sold" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">units_sold</th>
<th id="revenue" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">revenue</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="3" class="gt_group_heading">Computing</th>
</tr>

<tr>
<td class="gt_row gt_left gt_stub">Laptop</td>
<td class="gt_row gt_right">45</td>
<td class="gt_row gt_right">67,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Mouse</td>
<td class="gt_row gt_right">230</td>
<td class="gt_row gt_right">4,600</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Keyboard</td>
<td class="gt_row gt_right">180</td>
<td class="gt_row gt_right">9,000</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="3" class="gt_group_heading">Peripherals</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Monitor</td>
<td class="gt_row gt_right">65</td>
<td class="gt_row gt_right">19,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Webcam</td>
<td class="gt_row gt_right">120</td>
<td class="gt_row gt_right">6,000</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Headset</td>
<td class="gt_row gt_right">95</td>
<td class="gt_row gt_right">7,125</td>
</tr>
<tr>
<td class="gt_first_grand_summary_row_bottom gt_row gt_left gt_stub gt_grand_summary_row">Total</td>
<td class="gt_first_grand_summary_row_bottom gt_row gt_right gt_grand_summary_row">735</td>
<td class="gt_first_grand_summary_row_bottom gt_row gt_right gt_grand_summary_row">113725</td>
</tr>
</tbody>
</table>


The `numeric_only=True` argument ensures that only numeric columns are summed, avoiding errors with string columns.


# Styling Summary Rows

Summary rows can be styled using [loc.grand_summary()](../reference/loc.grand_summary.md#great_tables.loc.grand_summary) and `loc.summary()` with [tab_style()](../reference/GT.tab_style.md#great_tables.GT.tab_style). This lets you visually distinguish summary rows from data rows.


``` python
from great_tables import loc, style

(
    gt_sales
    .grand_summary_rows(
        fns={"Grand Total": pl.col("units_sold", "revenue").sum()}
    )
    .tab_style(
        style=style.fill(color="lightyellow"),
        locations=loc.grand_summary()
    )
)
```


<style>
#jqnbfculoq table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#jqnbfculoq thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#jqnbfculoq p { margin: 0; padding: 0; }
 #jqnbfculoq .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #jqnbfculoq .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #jqnbfculoq .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #jqnbfculoq .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #jqnbfculoq .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #jqnbfculoq .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #jqnbfculoq .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #jqnbfculoq .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #jqnbfculoq .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #jqnbfculoq .gt_column_spanner_outer:first-child { padding-left: 0; }
 #jqnbfculoq .gt_column_spanner_outer:last-child { padding-right: 0; }
 #jqnbfculoq .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #jqnbfculoq .gt_spanner_row { border-bottom-style: hidden; }
 #jqnbfculoq .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #jqnbfculoq .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #jqnbfculoq .gt_from_md> :first-child { margin-top: 0; }
 #jqnbfculoq .gt_from_md> :last-child { margin-bottom: 0; }
 #jqnbfculoq .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #jqnbfculoq .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #jqnbfculoq .gt_indent_1 { text-indent: 5px; }
 #jqnbfculoq .gt_indent_2 { text-indent: calc(5px * 2); }
 #jqnbfculoq .gt_indent_3 { text-indent: calc(5px * 3); }
 #jqnbfculoq .gt_indent_4 { text-indent: calc(5px * 4); }
 #jqnbfculoq .gt_indent_5 { text-indent: calc(5px * 5); }
 #jqnbfculoq .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #jqnbfculoq .gt_row_group_first td { border-top-width: 2px; }
 #jqnbfculoq .gt_row_group_first th { border-top-width: 2px; }
 #jqnbfculoq .gt_striped { color: #333333; background-color: #F4F4F4; }
 #jqnbfculoq .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #jqnbfculoq .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #jqnbfculoq .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #jqnbfculoq .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #jqnbfculoq .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #jqnbfculoq .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #jqnbfculoq .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #jqnbfculoq .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #jqnbfculoq .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #jqnbfculoq .gt_left { text-align: left; }
 #jqnbfculoq .gt_center { text-align: center; }
 #jqnbfculoq .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #jqnbfculoq .gt_font_normal { font-weight: normal; }
 #jqnbfculoq .gt_font_bold { font-weight: bold; }
 #jqnbfculoq .gt_font_italic { font-style: italic; }
 #jqnbfculoq .gt_super { font-size: 65%; }
 #jqnbfculoq .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #jqnbfculoq .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #jqnbfculoq .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #jqnbfculoq .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #jqnbfculoq .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #jqnbfculoq .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_title gt_font_normal">Q4 Product Sales</th>
</tr>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">By category</th>
</tr>
<tr class="gt_col_headings">
<th class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="units_sold" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">units_sold</th>
<th id="revenue" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">revenue</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="3" class="gt_group_heading">Computing</th>
</tr>

<tr>
<td class="gt_row gt_left gt_stub">Laptop</td>
<td class="gt_row gt_right">45</td>
<td class="gt_row gt_right">67,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Mouse</td>
<td class="gt_row gt_right">230</td>
<td class="gt_row gt_right">4,600</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Keyboard</td>
<td class="gt_row gt_right">180</td>
<td class="gt_row gt_right">9,000</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="3" class="gt_group_heading">Peripherals</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Monitor</td>
<td class="gt_row gt_right">65</td>
<td class="gt_row gt_right">19,500</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Webcam</td>
<td class="gt_row gt_right">120</td>
<td class="gt_row gt_right">6,000</td>
</tr>
<tr>
<td class="gt_row gt_left gt_stub">Headset</td>
<td class="gt_row gt_right">95</td>
<td class="gt_row gt_right">7,125</td>
</tr>
<tr>
<td class="gt_first_grand_summary_row_bottom gt_row gt_left gt_stub gt_grand_summary_row">Grand Total</td>
<td class="gt_first_grand_summary_row_bottom gt_row gt_right gt_grand_summary_row" style="background-color: lightyellow">735</td>
<td class="gt_first_grand_summary_row_bottom gt_row gt_right gt_grand_summary_row" style="background-color: lightyellow">113725</td>
</tr>
</tbody>
</table>


Summary rows are a natural companion to row groups, providing the aggregated context that readers need to interpret grouped data. By combining group-level and grand summaries, formatting, and targeted styling, you can build tables that tell a complete analytical story.
