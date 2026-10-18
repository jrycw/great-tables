# Footnotes

Footnotes provide a way to annotate specific cells, columns, or other table parts with additional context without cluttering the main display. **Great Tables** manages footnotes as a system: marks are automatically sequenced, placed consistently, and matched to their explanatory text in the table footer.

Footnotes are best suited for cell-specific or column-specific context that would clutter the main display if placed inline. For information that applies to the entire table, a source note in the footer (via [tab_source_note()](../reference/GT.tab_source_note.md#great_tables.GT.tab_source_note)) is usually a better fit because it avoids attaching a mark to any one location. And for systematic column descriptions, such as units or definitions, consider relabeling the column itself so the context is always visible without requiring the reader to look down at the footer.


# A Basic Footnote

Adding a footnote requires two things: the footnote text and a location specifier indicating where the footnote mark should appear. The [tab_footnote()](../reference/GT.tab_footnote.md#great_tables.GT.tab_footnote) method handles both, placing the mark at the targeted location and appending the text to the footer.


``` python
from great_tables import GT, md, loc
from great_tables.data import gibraltar

weather_mini = gibraltar.head(5)[["date", "time", "temp", "humidity", "wind_speed", "pressure"]]

(
    GT(weather_mini)
    .tab_header(title="Weather at Gibraltar Airport", subtitle="Observations on May 1, 2023")
    .tab_footnote(
        footnote="Air temperature in degrees Celsius.",
        locations=loc.column_labels(columns="temp")
    )
)
```


<style>
#jffqereowc table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#jffqereowc thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#jffqereowc p { margin: 0; padding: 0; }
 #jffqereowc .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #jffqereowc .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #jffqereowc .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #jffqereowc .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #jffqereowc .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #jffqereowc .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #jffqereowc .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #jffqereowc .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #jffqereowc .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #jffqereowc .gt_column_spanner_outer:first-child { padding-left: 0; }
 #jffqereowc .gt_column_spanner_outer:last-child { padding-right: 0; }
 #jffqereowc .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #jffqereowc .gt_spanner_row { border-bottom-style: hidden; }
 #jffqereowc .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #jffqereowc .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #jffqereowc .gt_from_md> :first-child { margin-top: 0; }
 #jffqereowc .gt_from_md> :last-child { margin-bottom: 0; }
 #jffqereowc .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #jffqereowc .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #jffqereowc .gt_indent_1 { text-indent: 5px; }
 #jffqereowc .gt_indent_2 { text-indent: calc(5px * 2); }
 #jffqereowc .gt_indent_3 { text-indent: calc(5px * 3); }
 #jffqereowc .gt_indent_4 { text-indent: calc(5px * 4); }
 #jffqereowc .gt_indent_5 { text-indent: calc(5px * 5); }
 #jffqereowc .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #jffqereowc .gt_row_group_first td { border-top-width: 2px; }
 #jffqereowc .gt_row_group_first th { border-top-width: 2px; }
 #jffqereowc .gt_striped { color: #333333; background-color: #F4F4F4; }
 #jffqereowc .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #jffqereowc .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #jffqereowc .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #jffqereowc .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #jffqereowc .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #jffqereowc .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #jffqereowc .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #jffqereowc .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #jffqereowc .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #jffqereowc .gt_left { text-align: left; }
 #jffqereowc .gt_center { text-align: center; }
 #jffqereowc .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #jffqereowc .gt_font_normal { font-weight: normal; }
 #jffqereowc .gt_font_bold { font-weight: bold; }
 #jffqereowc .gt_font_italic { font-style: italic; }
 #jffqereowc .gt_super { font-size: 65%; }
 #jffqereowc .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #jffqereowc .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #jffqereowc .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #jffqereowc .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #jffqereowc .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #jffqereowc .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="6" id="title" class="gt_heading gt_title gt_font_normal">Weather at Gibraltar Airport</th>
</tr>
<tr class="gt_heading">
<th colspan="6" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Observations on May 1, 2023</th>
</tr>
<tr class="gt_col_headings">
<th id="date" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">date</th>
<th id="time" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">time</th>
<th id="temp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">temp<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">1</span></th>
<th id="humidity" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">humidity</th>
<th id="wind_speed" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">wind_speed</th>
<th id="pressure" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">pressure</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">7.2</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:20</td>
<td class="gt_row gt_right" headers="temp">17.8</td>
<td class="gt_row gt_right" headers="humidity">0.77</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">02:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
</tbody><tfoot>
<tr class="gt_footnotes">
<td colspan="6" class="gt_footnote"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">1</span> Air temperature in degrees Celsius.</td>
</tr>
</tfoot>

</table>


The footnote mark (a superscript number) appears next to the `"temp"` column label, and the corresponding text appears at the bottom of the table.


# Targeting Different Locations

Footnotes can be attached to many different parts of the table. The `locations=` argument accepts any of the `loc` specifiers that support footnotes. Here are some of the most common targets.


## Column Labels

Attaching a footnote to a column label is useful for clarifying units or methodology.


``` python
(
    GT(weather_mini)
    .tab_header(title="Weather at Gibraltar Airport", subtitle="Observations on May 1, 2023")
    .tab_footnote(
        footnote="Relative humidity, as a value between 0 and 1.",
        locations=loc.column_labels(columns="humidity")
    )
)
```


<style>
#oogtjalmvj table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#oogtjalmvj thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#oogtjalmvj p { margin: 0; padding: 0; }
 #oogtjalmvj .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #oogtjalmvj .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #oogtjalmvj .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #oogtjalmvj .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #oogtjalmvj .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #oogtjalmvj .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #oogtjalmvj .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #oogtjalmvj .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #oogtjalmvj .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #oogtjalmvj .gt_column_spanner_outer:first-child { padding-left: 0; }
 #oogtjalmvj .gt_column_spanner_outer:last-child { padding-right: 0; }
 #oogtjalmvj .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #oogtjalmvj .gt_spanner_row { border-bottom-style: hidden; }
 #oogtjalmvj .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #oogtjalmvj .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #oogtjalmvj .gt_from_md> :first-child { margin-top: 0; }
 #oogtjalmvj .gt_from_md> :last-child { margin-bottom: 0; }
 #oogtjalmvj .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #oogtjalmvj .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #oogtjalmvj .gt_indent_1 { text-indent: 5px; }
 #oogtjalmvj .gt_indent_2 { text-indent: calc(5px * 2); }
 #oogtjalmvj .gt_indent_3 { text-indent: calc(5px * 3); }
 #oogtjalmvj .gt_indent_4 { text-indent: calc(5px * 4); }
 #oogtjalmvj .gt_indent_5 { text-indent: calc(5px * 5); }
 #oogtjalmvj .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #oogtjalmvj .gt_row_group_first td { border-top-width: 2px; }
 #oogtjalmvj .gt_row_group_first th { border-top-width: 2px; }
 #oogtjalmvj .gt_striped { color: #333333; background-color: #F4F4F4; }
 #oogtjalmvj .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #oogtjalmvj .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #oogtjalmvj .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #oogtjalmvj .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #oogtjalmvj .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #oogtjalmvj .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #oogtjalmvj .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #oogtjalmvj .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #oogtjalmvj .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #oogtjalmvj .gt_left { text-align: left; }
 #oogtjalmvj .gt_center { text-align: center; }
 #oogtjalmvj .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #oogtjalmvj .gt_font_normal { font-weight: normal; }
 #oogtjalmvj .gt_font_bold { font-weight: bold; }
 #oogtjalmvj .gt_font_italic { font-style: italic; }
 #oogtjalmvj .gt_super { font-size: 65%; }
 #oogtjalmvj .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #oogtjalmvj .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #oogtjalmvj .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #oogtjalmvj .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #oogtjalmvj .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #oogtjalmvj .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="6" id="title" class="gt_heading gt_title gt_font_normal">Weather at Gibraltar Airport</th>
</tr>
<tr class="gt_heading">
<th colspan="6" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Observations on May 1, 2023</th>
</tr>
<tr class="gt_col_headings">
<th id="date" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">date</th>
<th id="time" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">time</th>
<th id="temp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">temp</th>
<th id="humidity" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">humidity<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">1</span></th>
<th id="wind_speed" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">wind_speed</th>
<th id="pressure" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">pressure</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">7.2</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:20</td>
<td class="gt_row gt_right" headers="temp">17.8</td>
<td class="gt_row gt_right" headers="humidity">0.77</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">02:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
</tbody><tfoot>
<tr class="gt_footnotes">
<td colspan="6" class="gt_footnote"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">1</span> Relative humidity, as a value between 0 and 1.</td>
</tr>
</tfoot>

</table>


## Body Cells

You can annotate specific data cells by targeting them with [loc.body()](../reference/loc.body.md#great_tables.loc.body).


``` python
(
    GT(weather_mini)
    .tab_header(title="Weather at Gibraltar Airport", subtitle="Observations on May 1, 2023")
    .tab_footnote(
        footnote="Lowest temperature in this sample.",
        locations=loc.body(columns="temp", rows=[2])
    )
)
```


<style>
#cwxsuusvqq table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#cwxsuusvqq thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#cwxsuusvqq p { margin: 0; padding: 0; }
 #cwxsuusvqq .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #cwxsuusvqq .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #cwxsuusvqq .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #cwxsuusvqq .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #cwxsuusvqq .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #cwxsuusvqq .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #cwxsuusvqq .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #cwxsuusvqq .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #cwxsuusvqq .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #cwxsuusvqq .gt_column_spanner_outer:first-child { padding-left: 0; }
 #cwxsuusvqq .gt_column_spanner_outer:last-child { padding-right: 0; }
 #cwxsuusvqq .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #cwxsuusvqq .gt_spanner_row { border-bottom-style: hidden; }
 #cwxsuusvqq .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #cwxsuusvqq .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #cwxsuusvqq .gt_from_md> :first-child { margin-top: 0; }
 #cwxsuusvqq .gt_from_md> :last-child { margin-bottom: 0; }
 #cwxsuusvqq .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #cwxsuusvqq .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #cwxsuusvqq .gt_indent_1 { text-indent: 5px; }
 #cwxsuusvqq .gt_indent_2 { text-indent: calc(5px * 2); }
 #cwxsuusvqq .gt_indent_3 { text-indent: calc(5px * 3); }
 #cwxsuusvqq .gt_indent_4 { text-indent: calc(5px * 4); }
 #cwxsuusvqq .gt_indent_5 { text-indent: calc(5px * 5); }
 #cwxsuusvqq .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #cwxsuusvqq .gt_row_group_first td { border-top-width: 2px; }
 #cwxsuusvqq .gt_row_group_first th { border-top-width: 2px; }
 #cwxsuusvqq .gt_striped { color: #333333; background-color: #F4F4F4; }
 #cwxsuusvqq .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #cwxsuusvqq .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #cwxsuusvqq .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #cwxsuusvqq .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #cwxsuusvqq .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #cwxsuusvqq .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #cwxsuusvqq .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #cwxsuusvqq .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #cwxsuusvqq .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #cwxsuusvqq .gt_left { text-align: left; }
 #cwxsuusvqq .gt_center { text-align: center; }
 #cwxsuusvqq .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #cwxsuusvqq .gt_font_normal { font-weight: normal; }
 #cwxsuusvqq .gt_font_bold { font-weight: bold; }
 #cwxsuusvqq .gt_font_italic { font-style: italic; }
 #cwxsuusvqq .gt_super { font-size: 65%; }
 #cwxsuusvqq .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #cwxsuusvqq .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #cwxsuusvqq .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #cwxsuusvqq .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #cwxsuusvqq .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #cwxsuusvqq .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="6" id="title" class="gt_heading gt_title gt_font_normal">Weather at Gibraltar Airport</th>
</tr>
<tr class="gt_heading">
<th colspan="6" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Observations on May 1, 2023</th>
</tr>
<tr class="gt_col_headings">
<th id="date" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">date</th>
<th id="time" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">time</th>
<th id="temp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">temp</th>
<th id="humidity" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">humidity</th>
<th id="wind_speed" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">wind_speed</th>
<th id="pressure" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">pressure</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">7.2</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:20</td>
<td class="gt_row gt_right" headers="temp"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">1</span> 17.8</td>
<td class="gt_row gt_right" headers="humidity">0.77</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">02:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
</tbody><tfoot>
<tr class="gt_footnotes">
<td colspan="6" class="gt_footnote"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">1</span> Lowest temperature in this sample.</td>
</tr>
</tfoot>

</table>


## The Title or Subtitle

Footnotes on the table header can provide methodological notes or data source context.


``` python
(
    GT(weather_mini)
    .tab_header(title="Weather at Gibraltar Airport", subtitle="Observations on May 1, 2023")
    .tab_footnote(
        footnote="Data collected at the airport's weather station, 5 m above mean sea level.",
        locations=loc.title()
    )
)
```


<style>
#mzurnvlhac table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#mzurnvlhac thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#mzurnvlhac p { margin: 0; padding: 0; }
 #mzurnvlhac .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #mzurnvlhac .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #mzurnvlhac .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #mzurnvlhac .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #mzurnvlhac .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #mzurnvlhac .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #mzurnvlhac .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #mzurnvlhac .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #mzurnvlhac .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #mzurnvlhac .gt_column_spanner_outer:first-child { padding-left: 0; }
 #mzurnvlhac .gt_column_spanner_outer:last-child { padding-right: 0; }
 #mzurnvlhac .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #mzurnvlhac .gt_spanner_row { border-bottom-style: hidden; }
 #mzurnvlhac .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #mzurnvlhac .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #mzurnvlhac .gt_from_md> :first-child { margin-top: 0; }
 #mzurnvlhac .gt_from_md> :last-child { margin-bottom: 0; }
 #mzurnvlhac .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #mzurnvlhac .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #mzurnvlhac .gt_indent_1 { text-indent: 5px; }
 #mzurnvlhac .gt_indent_2 { text-indent: calc(5px * 2); }
 #mzurnvlhac .gt_indent_3 { text-indent: calc(5px * 3); }
 #mzurnvlhac .gt_indent_4 { text-indent: calc(5px * 4); }
 #mzurnvlhac .gt_indent_5 { text-indent: calc(5px * 5); }
 #mzurnvlhac .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #mzurnvlhac .gt_row_group_first td { border-top-width: 2px; }
 #mzurnvlhac .gt_row_group_first th { border-top-width: 2px; }
 #mzurnvlhac .gt_striped { color: #333333; background-color: #F4F4F4; }
 #mzurnvlhac .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #mzurnvlhac .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #mzurnvlhac .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #mzurnvlhac .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #mzurnvlhac .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #mzurnvlhac .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #mzurnvlhac .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #mzurnvlhac .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #mzurnvlhac .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #mzurnvlhac .gt_left { text-align: left; }
 #mzurnvlhac .gt_center { text-align: center; }
 #mzurnvlhac .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #mzurnvlhac .gt_font_normal { font-weight: normal; }
 #mzurnvlhac .gt_font_bold { font-weight: bold; }
 #mzurnvlhac .gt_font_italic { font-style: italic; }
 #mzurnvlhac .gt_super { font-size: 65%; }
 #mzurnvlhac .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #mzurnvlhac .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #mzurnvlhac .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #mzurnvlhac .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #mzurnvlhac .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #mzurnvlhac .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="6" id="title" class="gt_heading gt_title gt_font_normal">Weather at Gibraltar Airport<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">1</span></th>
</tr>
<tr class="gt_heading">
<th colspan="6" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Observations on May 1, 2023</th>
</tr>
<tr class="gt_col_headings">
<th id="date" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">date</th>
<th id="time" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">time</th>
<th id="temp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">temp</th>
<th id="humidity" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">humidity</th>
<th id="wind_speed" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">wind_speed</th>
<th id="pressure" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">pressure</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">7.2</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:20</td>
<td class="gt_row gt_right" headers="temp">17.8</td>
<td class="gt_row gt_right" headers="humidity">0.77</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">02:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
</tbody><tfoot>
<tr class="gt_footnotes">
<td colspan="6" class="gt_footnote"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">1</span> Data collected at the airport's weather station, 5 m above mean sea level.</td>
</tr>
</tfoot>

</table>


# Multiple Footnotes

You can add multiple footnotes to a single table. Each call to [tab_footnote()](../reference/GT.tab_footnote.md#great_tables.GT.tab_footnote) creates a new footnote with its own sequenced mark. The marks are numbered in the order they appear in the table (reading left-to-right, top-to-bottom).


``` python
(
    GT(weather_mini)
    .tab_header(title="Weather at Gibraltar Airport", subtitle="Observations on May 1, 2023")
    .tab_footnote(
        footnote="Air temperature in degrees Celsius.",
        locations=loc.column_labels(columns="temp")
    )
    .tab_footnote(
        footnote="Atmospheric pressure in hectopascals.",
        locations=loc.column_labels(columns="pressure")
    )
    .tab_footnote(
        footnote="Highest wind speed in this sample.",
        locations=loc.body(columns="wind_speed", rows=[1])
    )
)
```


<style>
#jdwwhffxbm table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#jdwwhffxbm thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#jdwwhffxbm p { margin: 0; padding: 0; }
 #jdwwhffxbm .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #jdwwhffxbm .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #jdwwhffxbm .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #jdwwhffxbm .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #jdwwhffxbm .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #jdwwhffxbm .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #jdwwhffxbm .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #jdwwhffxbm .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #jdwwhffxbm .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #jdwwhffxbm .gt_column_spanner_outer:first-child { padding-left: 0; }
 #jdwwhffxbm .gt_column_spanner_outer:last-child { padding-right: 0; }
 #jdwwhffxbm .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #jdwwhffxbm .gt_spanner_row { border-bottom-style: hidden; }
 #jdwwhffxbm .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #jdwwhffxbm .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #jdwwhffxbm .gt_from_md> :first-child { margin-top: 0; }
 #jdwwhffxbm .gt_from_md> :last-child { margin-bottom: 0; }
 #jdwwhffxbm .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #jdwwhffxbm .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #jdwwhffxbm .gt_indent_1 { text-indent: 5px; }
 #jdwwhffxbm .gt_indent_2 { text-indent: calc(5px * 2); }
 #jdwwhffxbm .gt_indent_3 { text-indent: calc(5px * 3); }
 #jdwwhffxbm .gt_indent_4 { text-indent: calc(5px * 4); }
 #jdwwhffxbm .gt_indent_5 { text-indent: calc(5px * 5); }
 #jdwwhffxbm .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #jdwwhffxbm .gt_row_group_first td { border-top-width: 2px; }
 #jdwwhffxbm .gt_row_group_first th { border-top-width: 2px; }
 #jdwwhffxbm .gt_striped { color: #333333; background-color: #F4F4F4; }
 #jdwwhffxbm .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #jdwwhffxbm .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #jdwwhffxbm .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #jdwwhffxbm .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #jdwwhffxbm .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #jdwwhffxbm .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #jdwwhffxbm .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #jdwwhffxbm .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #jdwwhffxbm .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #jdwwhffxbm .gt_left { text-align: left; }
 #jdwwhffxbm .gt_center { text-align: center; }
 #jdwwhffxbm .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #jdwwhffxbm .gt_font_normal { font-weight: normal; }
 #jdwwhffxbm .gt_font_bold { font-weight: bold; }
 #jdwwhffxbm .gt_font_italic { font-style: italic; }
 #jdwwhffxbm .gt_super { font-size: 65%; }
 #jdwwhffxbm .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #jdwwhffxbm .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #jdwwhffxbm .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #jdwwhffxbm .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #jdwwhffxbm .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #jdwwhffxbm .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="6" id="title" class="gt_heading gt_title gt_font_normal">Weather at Gibraltar Airport</th>
</tr>
<tr class="gt_heading">
<th colspan="6" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Observations on May 1, 2023</th>
</tr>
<tr class="gt_col_headings">
<th id="date" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">date</th>
<th id="time" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">time</th>
<th id="temp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">temp<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">1</span></th>
<th id="humidity" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">humidity</th>
<th id="wind_speed" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">wind_speed</th>
<th id="pressure" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">pressure<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">2</span></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">3</span> 7.2</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:20</td>
<td class="gt_row gt_right" headers="temp">17.8</td>
<td class="gt_row gt_right" headers="humidity">0.77</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">02:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
</tbody><tfoot>
<tr class="gt_footnotes">
<td colspan="6" class="gt_footnote"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">1</span> Air temperature in degrees Celsius.</td>
</tr>
<tr class="gt_footnotes">
<td colspan="6" class="gt_footnote"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">2</span> Atmospheric pressure in hectopascals.</td>
</tr>
<tr class="gt_footnotes">
<td colspan="6" class="gt_footnote"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">3</span> Highest wind speed in this sample.</td>
</tr>
</tfoot>

</table>


Three footnote marks are placed in the table, and three corresponding notes appear in the footer, each with its sequential number.

While you can add many footnotes to a single table, use restraint. More than three or four footnotes can overwhelm the reader and make the footer itself difficult to navigate. If you find yourself adding footnotes to most cells, consider whether the information would be better conveyed through more descriptive column labels, a subtitle, or a separate explanatory paragraph outside the table.


# Footnotes Without a Mark

If you want to include explanatory text in the footer without attaching a mark to any cell, omit the `locations=` argument (or set it to `None`).


``` python
(
    GT(weather_mini)
    .tab_header(title="Weather at Gibraltar Airport", subtitle="Observations on May 1, 2023")
    .tab_footnote(footnote="All observations are from Gibraltar Airport, May 2023.")
)
```


<style>
#sznhalecyv table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#sznhalecyv thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#sznhalecyv p { margin: 0; padding: 0; }
 #sznhalecyv .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #sznhalecyv .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #sznhalecyv .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #sznhalecyv .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #sznhalecyv .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #sznhalecyv .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #sznhalecyv .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #sznhalecyv .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #sznhalecyv .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #sznhalecyv .gt_column_spanner_outer:first-child { padding-left: 0; }
 #sznhalecyv .gt_column_spanner_outer:last-child { padding-right: 0; }
 #sznhalecyv .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #sznhalecyv .gt_spanner_row { border-bottom-style: hidden; }
 #sznhalecyv .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #sznhalecyv .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #sznhalecyv .gt_from_md> :first-child { margin-top: 0; }
 #sznhalecyv .gt_from_md> :last-child { margin-bottom: 0; }
 #sznhalecyv .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #sznhalecyv .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #sznhalecyv .gt_indent_1 { text-indent: 5px; }
 #sznhalecyv .gt_indent_2 { text-indent: calc(5px * 2); }
 #sznhalecyv .gt_indent_3 { text-indent: calc(5px * 3); }
 #sznhalecyv .gt_indent_4 { text-indent: calc(5px * 4); }
 #sznhalecyv .gt_indent_5 { text-indent: calc(5px * 5); }
 #sznhalecyv .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #sznhalecyv .gt_row_group_first td { border-top-width: 2px; }
 #sznhalecyv .gt_row_group_first th { border-top-width: 2px; }
 #sznhalecyv .gt_striped { color: #333333; background-color: #F4F4F4; }
 #sznhalecyv .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #sznhalecyv .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #sznhalecyv .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #sznhalecyv .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #sznhalecyv .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #sznhalecyv .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #sznhalecyv .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #sznhalecyv .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #sznhalecyv .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #sznhalecyv .gt_left { text-align: left; }
 #sznhalecyv .gt_center { text-align: center; }
 #sznhalecyv .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #sznhalecyv .gt_font_normal { font-weight: normal; }
 #sznhalecyv .gt_font_bold { font-weight: bold; }
 #sznhalecyv .gt_font_italic { font-style: italic; }
 #sznhalecyv .gt_super { font-size: 65%; }
 #sznhalecyv .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #sznhalecyv .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #sznhalecyv .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #sznhalecyv .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #sznhalecyv .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #sznhalecyv .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="6" id="title" class="gt_heading gt_title gt_font_normal">Weather at Gibraltar Airport</th>
</tr>
<tr class="gt_heading">
<th colspan="6" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Observations on May 1, 2023</th>
</tr>
<tr class="gt_col_headings">
<th id="date" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">date</th>
<th id="time" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">time</th>
<th id="temp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">temp</th>
<th id="humidity" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">humidity</th>
<th id="wind_speed" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">wind_speed</th>
<th id="pressure" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">pressure</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">7.2</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:20</td>
<td class="gt_row gt_right" headers="temp">17.8</td>
<td class="gt_row gt_right" headers="humidity">0.77</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">02:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
</tbody><tfoot>
<tr class="gt_footnotes">
<td colspan="6" class="gt_footnote">All observations are from Gibraltar Airport, May 2023.</td>
</tr>
</tfoot>

</table>


This is useful for general notes that apply to the entire table rather than to a specific cell or label. Markless footnotes work well for disclaimers, methodological notes, or data-source attributions that apply broadly rather than to any single cell. Because they carry no superscript mark, they read as supplementary context for the table as a whole.


# Controlling Mark Placement

The `placement=` argument determines where the footnote mark appears relative to the cell content. The options are `"auto"` (the default), `"left"`, and `"right"`.


``` python
(
    GT(weather_mini)
    .tab_header(title="Weather at Gibraltar Airport", subtitle="Observations on May 1, 2023")
    .tab_footnote(
        footnote="Wind speed in meters per second.",
        locations=loc.column_labels(columns="wind_speed"),
        placement="left"
    )
)
```


<style>
#dragbrsdwf table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#dragbrsdwf thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#dragbrsdwf p { margin: 0; padding: 0; }
 #dragbrsdwf .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #dragbrsdwf .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #dragbrsdwf .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #dragbrsdwf .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #dragbrsdwf .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #dragbrsdwf .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #dragbrsdwf .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #dragbrsdwf .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #dragbrsdwf .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #dragbrsdwf .gt_column_spanner_outer:first-child { padding-left: 0; }
 #dragbrsdwf .gt_column_spanner_outer:last-child { padding-right: 0; }
 #dragbrsdwf .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #dragbrsdwf .gt_spanner_row { border-bottom-style: hidden; }
 #dragbrsdwf .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #dragbrsdwf .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #dragbrsdwf .gt_from_md> :first-child { margin-top: 0; }
 #dragbrsdwf .gt_from_md> :last-child { margin-bottom: 0; }
 #dragbrsdwf .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #dragbrsdwf .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #dragbrsdwf .gt_indent_1 { text-indent: 5px; }
 #dragbrsdwf .gt_indent_2 { text-indent: calc(5px * 2); }
 #dragbrsdwf .gt_indent_3 { text-indent: calc(5px * 3); }
 #dragbrsdwf .gt_indent_4 { text-indent: calc(5px * 4); }
 #dragbrsdwf .gt_indent_5 { text-indent: calc(5px * 5); }
 #dragbrsdwf .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #dragbrsdwf .gt_row_group_first td { border-top-width: 2px; }
 #dragbrsdwf .gt_row_group_first th { border-top-width: 2px; }
 #dragbrsdwf .gt_striped { color: #333333; background-color: #F4F4F4; }
 #dragbrsdwf .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #dragbrsdwf .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #dragbrsdwf .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #dragbrsdwf .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #dragbrsdwf .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #dragbrsdwf .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #dragbrsdwf .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #dragbrsdwf .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #dragbrsdwf .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #dragbrsdwf .gt_left { text-align: left; }
 #dragbrsdwf .gt_center { text-align: center; }
 #dragbrsdwf .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #dragbrsdwf .gt_font_normal { font-weight: normal; }
 #dragbrsdwf .gt_font_bold { font-weight: bold; }
 #dragbrsdwf .gt_font_italic { font-style: italic; }
 #dragbrsdwf .gt_super { font-size: 65%; }
 #dragbrsdwf .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #dragbrsdwf .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #dragbrsdwf .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #dragbrsdwf .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #dragbrsdwf .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #dragbrsdwf .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="6" id="title" class="gt_heading gt_title gt_font_normal">Weather at Gibraltar Airport</th>
</tr>
<tr class="gt_heading">
<th colspan="6" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Observations on May 1, 2023</th>
</tr>
<tr class="gt_col_headings">
<th id="date" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">date</th>
<th id="time" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">time</th>
<th id="temp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">temp</th>
<th id="humidity" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">humidity</th>
<th id="wind_speed" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">1</span> wind_speed</th>
<th id="pressure" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">pressure</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">7.2</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:20</td>
<td class="gt_row gt_right" headers="temp">17.8</td>
<td class="gt_row gt_right" headers="humidity">0.77</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">02:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
</tbody><tfoot>
<tr class="gt_footnotes">
<td colspan="6" class="gt_footnote"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">1</span> Wind speed in meters per second.</td>
</tr>
</tfoot>

</table>


With `placement="left"`, the footnote mark appears before the cell text rather than after it.


# Using Markdown in Footnotes

Footnote text supports Markdown formatting through the [md()](../reference/md.md#great_tables.md) helper function. This lets you include emphasis, links, or other inline formatting in your footnotes.


``` python
(
    GT(weather_mini)
    .tab_header(title="Weather at Gibraltar Airport", subtitle="Observations on May 1, 2023")
    .tab_footnote(
        footnote=md("Observations from the *Gibraltar Airport Station* (GIB)."),
        locations=loc.title()
    )
)
```


<style>
#mwagqovalq table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#mwagqovalq thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#mwagqovalq p { margin: 0; padding: 0; }
 #mwagqovalq .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #mwagqovalq .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #mwagqovalq .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #mwagqovalq .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #mwagqovalq .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #mwagqovalq .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #mwagqovalq .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #mwagqovalq .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #mwagqovalq .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #mwagqovalq .gt_column_spanner_outer:first-child { padding-left: 0; }
 #mwagqovalq .gt_column_spanner_outer:last-child { padding-right: 0; }
 #mwagqovalq .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #mwagqovalq .gt_spanner_row { border-bottom-style: hidden; }
 #mwagqovalq .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #mwagqovalq .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #mwagqovalq .gt_from_md> :first-child { margin-top: 0; }
 #mwagqovalq .gt_from_md> :last-child { margin-bottom: 0; }
 #mwagqovalq .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #mwagqovalq .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #mwagqovalq .gt_indent_1 { text-indent: 5px; }
 #mwagqovalq .gt_indent_2 { text-indent: calc(5px * 2); }
 #mwagqovalq .gt_indent_3 { text-indent: calc(5px * 3); }
 #mwagqovalq .gt_indent_4 { text-indent: calc(5px * 4); }
 #mwagqovalq .gt_indent_5 { text-indent: calc(5px * 5); }
 #mwagqovalq .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #mwagqovalq .gt_row_group_first td { border-top-width: 2px; }
 #mwagqovalq .gt_row_group_first th { border-top-width: 2px; }
 #mwagqovalq .gt_striped { color: #333333; background-color: #F4F4F4; }
 #mwagqovalq .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #mwagqovalq .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #mwagqovalq .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #mwagqovalq .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #mwagqovalq .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #mwagqovalq .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #mwagqovalq .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #mwagqovalq .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #mwagqovalq .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #mwagqovalq .gt_left { text-align: left; }
 #mwagqovalq .gt_center { text-align: center; }
 #mwagqovalq .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #mwagqovalq .gt_font_normal { font-weight: normal; }
 #mwagqovalq .gt_font_bold { font-weight: bold; }
 #mwagqovalq .gt_font_italic { font-style: italic; }
 #mwagqovalq .gt_super { font-size: 65%; }
 #mwagqovalq .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #mwagqovalq .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #mwagqovalq .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #mwagqovalq .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #mwagqovalq .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #mwagqovalq .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="6" id="title" class="gt_heading gt_title gt_font_normal">Weather at Gibraltar Airport<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">1</span></th>
</tr>
<tr class="gt_heading">
<th colspan="6" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Observations on May 1, 2023</th>
</tr>
<tr class="gt_col_headings">
<th id="date" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">date</th>
<th id="time" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">time</th>
<th id="temp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">temp</th>
<th id="humidity" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">humidity</th>
<th id="wind_speed" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">wind_speed</th>
<th id="pressure" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">pressure</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">00:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">7.2</td>
<td class="gt_row gt_right" headers="pressure">1015.2</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:20</td>
<td class="gt_row gt_right" headers="temp">17.8</td>
<td class="gt_row gt_right" headers="humidity">0.77</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">01:50</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.73</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
<tr>
<td class="gt_row gt_right" headers="date">2023-05-01</td>
<td class="gt_row gt_right" headers="time">02:20</td>
<td class="gt_row gt_right" headers="temp">18.9</td>
<td class="gt_row gt_right" headers="humidity">0.68</td>
<td class="gt_row gt_right" headers="wind_speed">6.7</td>
<td class="gt_row gt_right" headers="pressure">1014.6</td>
</tr>
</tbody><tfoot>
<tr class="gt_footnotes">
<td colspan="6" class="gt_footnote"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;line-height:0;">1</span> Observations from the <em>Gibraltar Airport Station</em> (GIB).</td>
</tr>
</tfoot>

</table>


Footnotes are a subtle but important tool for building informative tables. They let you add precision and context where it matters most, keeping the main table body clean while ensuring readers have access to the details they need. The automatic sequencing and placement system means you can focus on content rather than managing mark numbers manually.
