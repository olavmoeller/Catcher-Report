# Catcher Report

## Overview
This project will generate a report on catchers from a CSV file, which will give a summary of their framing abilities versus each hand of hitter, their throw speed, and their manually inputted caught stealing rates.

## Usage
1) Select the team you're using the script on, for template selection.
2) Choose the date/folder name.
3) Add the catcher's name and stolen base data to the sb_att_dict in the format shown. You can remove the catchers that aren't relevant anymore, although it doens't really matter. This can be usually found on the Oregon State Baseball cumulative statistics website.
4) Run the script, and select the CSV when prompted.
5) Output will be saved to a created subfolder called output, then a subfolder of that named catcher_reports_{date}, where date is what you entered at the top. There will be a catcher plots folder, which just stores the framing graphics, and then the actual report folder will have an overall multi page pdf and ppt for every catcher in the CSV, and then in the PDF subfolder, it will have separate pdfs for each individual catcher.

## Credit
-Author: Olav Moeller