function drawNanCountChart() {
    let header = ['Feature', 'Missing Values Count']
    let result = Object.keys(nan_json_data).map((key) => [key, nan_json_data[key][0]]);
    result.unshift(header)
    console.log(result)
    var data = google.visualization.arrayToDataTable(result);

    var options = {
        title: 'Missing Values in different features of our dataset',
        hAxis: {
            title: 'Total Missing Values'
        },
        vAxis: {
            title: 'Feature'
        }
    };

    var chart = new google.visualization.BarChart(document.getElementById('chart_div'));
    chart.draw(data, options);
}


function drawNanPercentChart() {
    let header = ['Feature', 'Missing Values Percentage']
    let result = Object.keys(nan_json_data).map((key) => [key, nan_json_data[key][1]]);
    result.unshift(header)
    console.log(result)
    var data = google.visualization.arrayToDataTable(result);

    var options = {
        title: 'Missing Values in different features of our dataset',
        hAxis: {
            title: 'Total Missing Percentage',
            maxValue: 63
        },
        vAxis: {
            title: 'Feature'
        }
    };

    var chart = new google.visualization.BarChart(document.getElementById('chart_div1'));
    chart.draw(data, options);
}


function drawStateAccidentsChart() {
    let header = ['State', 'Accidents Count']
    let result = Object.keys(state_json_data).map((key) => [key, state_json_data[key]]);
    result.unshift(header)
    console.log(result)
    var data = google.visualization.arrayToDataTable(result);

    var options = {
        title: "Accidents Count per each state",
        region: 'US',
        displayMode: 'markers',
        colorAxis: {colors: ['#9d3b3b', 'black', '#e31b23']},
        datalessRegionColor: '#8ac4ad',
        defaultColor: '#f5f5f5',
    };

    var chart = new google.visualization.GeoChart(document.getElementById('chart_div2'));
    chart.draw(data, options);
}
