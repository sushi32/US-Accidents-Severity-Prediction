const chart1 = dc.barChart("#chart1")
const chart2 = dc.pieChart("#chart2")
const chart3 = dc.pieChart('#chart3')
const chart4 = dc.rowChart('#chart4')
const chart5 = dc.rowChart("#chart5")
const chart6 = dc.pieChart("#chart6")
const chart7 = dc.rowChart("#chart7")
const chart8 = dc.dataTable("#chart8")

// chrome.exe --user-data-dir="C://Chrome dev session" --disable-web-security
d3.csv('../static/data_csv/accidents_data.csv', function (data) {
    let dtgFormat = d3.time.format("%Y-%m-%dT%H:%M:%S");

    data.forEach(function (d) {
        d.date = d["Start_Time"]
        d.dtg = dtgFormat.parse(d.date);
        d.dtg1  = d.date.substr(0,10) + " " + d.date.substr(11,8);
        d.temperature = d["Temperature(F)"];
        d.hour = d["Hour"];
        d.side = d["Side"];
        d.severity = d["Severity"];
        d.weekday = d["Weekday"];
        d.state = d["State"];
        d.weather = d["Weather_Condition"];
        d.wind = d["Wind_Direction"];
        d.windch = d["Wind_Chill(F)"]
        d.humid = d["Humidity(%)"]
        d.pressure = d["Pressure(in)"]
        d.lat = d["Start_Lat"]
        d.lng = d["Start_Lng"]
        d.visibility = d["Visibility(mi)"]
        d.windspd = d["Wind_Speed(mph)"]
        d.dist = d["Distance(mi)"]
        d.cross = d["Crossing"]
        d.junct = d["Junction"]
        d.ts = d["Traffic_Signal"]
        d.sun = d["Sunrise_Sunset"]
    });

    const facts = crossfilter(data);
    const all = facts.groupAll();

    const timeDimension = facts.dimension(function (d) {
        return d.dtg;
    });

    const hourVal = facts.dimension(function (d) {
        return d.hour;
    });
    const hourGroup = hourVal.group();

    const sideVal = facts.dimension(function (d) {
        return d.side;
    });
    const sideValGroup = sideVal.group();

    const seval = facts.dimension(function (d) {
        return Math.floor(d.severity);
    });
    const seValGroupCount = seval.group();

    const dayOfWeek = facts.dimension(function (d) {
        const weekDayName = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
        return `${Math.floor(d.weekday)}.${weekDayName[Math.floor(d.weekday)]}`;
    })
    const dayOfWeekGroup = dayOfWeek.group();

    const stateVal = facts.dimension(function (d) {
        return d.state;
    })
    const stateValGroup = stateVal.group();

    const weatherVal = facts.dimension(function (d) {
        return d.weather;
    })
    const weatherValGroup = weatherVal.group();

    const windVal = facts.dimension(function (d) {
        return d.wind;
    })
    const windValGroup = windVal.group();

    chart1.width(400)
        .height(250)
        .margins({top: 10, right: 10, bottom: 20, left: 40})
        .dimension(hourVal)
        .group(hourGroup)
        .transitionDuration(500)
        .centerBar(true)
        .gap(70)
        .x(d3.scale.linear().domain([0, 24]))
        .elasticY(true)
        .xAxis().tickFormat();

    chart2
        .width(300)
        .height(300)
        .dimension(sideVal)
        .group(sideValGroup)
        .colors(d3.scale.category20b())
        .label(c => {
            let k = ""
            if (c.data.key === "L") {
                k = "Left"
            } else {
                k = "Right"
            }
            if (chart2.hasFilter() && !chart2.hasFilter(c.data.key)) {
                return `${k} (0%)`;
            }
            let label = k + " ";
            if (all.value()) {
                label += `(${Math.floor(c.value / all.value() * 100)}%)`;
            }
            return label;
        });

    chart3
        .width(300)
        .height(300)
        .dimension(seval)
        .group(seValGroupCount)
        .colors(d3.scale.category20b())
        .label(c => {
            if (chart3.hasFilter() && !chart3.hasFilter(c.data.key)) {
                return `${c.data.key} (0%)`;
            }
            let label = c.data.key + " ";
            if (all.value()) {
                label += `(${Math.floor(c.value / all.value() * 100)}%)`;
            }
            return label;
        });

    chart6
        .width(300)
        .height(300)
        .dimension(weatherVal)
        .group(weatherValGroup)
        .colors(d3.scale.category20b())
        .label(c => {
            console.log(c)
            let v = ""
            if (c.data.key === "0") {
                v = "Good"
            } else if (c.data.key === "1") {
                v = "Mild"
            } else {
                v = "Bad"
            }

            if (chart6.hasFilter() && !chart6.hasFilter(c.data.key)) {
                return `${v} (0%)`;
            }
            let label = v + " ";
            if (all.value()) {
                label += `(${Math.floor(c.value / all.value() * 100)}%)`;
            }
            return label;
        });

    chart4
        .width(550)
        .height(250)
        .margins({top: 20, left: 10, right: 10, bottom: 20})
        .group(dayOfWeekGroup)
        .dimension(dayOfWeek)
        .colors(d3.scale.category20())
        .label(c => c.key.split('.')[1])
        .title(c => c.value)
        .elasticX(true)
        .xAxis().ticks(4);

    chart5
        .width(700)
        .height(1150)
        .margins({top: 20, left: 10, right: 10, bottom: 20})
        .group(stateValGroup)
        .dimension(stateVal)
        .colors(d3.scale.category10())
        .label(c => c.key)
        .title(c => c.value)
        .elasticX(true)
        .xAxis().ticks(6);

    chart7
        .width(700)
        .height(750)
        .margins({top: 20, left: 10, right: 10, bottom: 20})
        .group(windValGroup)
        .dimension(windVal)
        .colors(d3.scale.category20())
        .label(c => c.key)
        .title(c => c.key + " WEATHER: " + c.value)
        .elasticX(true)
        .xAxis().ticks(6);

    chart8
        .dimension(timeDimension)
        .group(function (d) {
            return "Accidents Feature Values Data Table"
        })
        .size(13)
        .columns([
            function (d) {
                return d.dtg1;
            },
            function (d) {
                return d.temperature;
            },
            function (d) {
                return d.windch;
            },
            function (d) {
                return d.humid;
            },
            function (d) {
                return d.pressure;
            },
            function (d) {
                return d.visibility;
            },
            function (d) {
                return d.windspd;
            },
            function (d) {
                return d.dist;
            },
            function (d) {
                return d.cross;
            },
            function (d) {
                return d.junct;
            },
            function (d) {
                return d.ts;
            },
            function (d) {
                return d.sun;
            },
            function (d) {
                return '<a style="color: orange" href=\"http://maps.google.com/maps?z=12&t=m&q=loc:' + d.lat + '+' + d.lng + "\" target=\"_blank\">Google Map</a>"
            },
            function (d) {
                return '<a style="color: darkolivegreen" href=\"http://www.openstreetmap.org/?mlat=' + d.lat + '&mlon=' + d.lng +'&zoom=12'+ "\" target=\"_blank\"> OSM Map</a>"
            }
        ]);

    dc.renderAll();
});