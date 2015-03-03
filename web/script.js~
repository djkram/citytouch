var g_dataset = null;
var g_time = null;
var g_previous_point = null;
var g_pos = null;
var g_plot = null;
var g_tweet_group = L.featureGroup();
var g_item = null;
var g_tweetClouds = null;
var g_hashClouds = null;
var g_selected_cluster = null;
var show_series = [true, true, true];
var timeline2;

var colors = d3.scale.category10();
var wkt = new Wkt.Wkt();

var svg = d3.select("#word_cloud").append("svg")
    .attr("width", 500)
    .attr("height", 500)

var background = svg.append("g");

var vis = svg.append("g")
    .attr("transform", "translate(250, 250)")

var fill = d3.scale.category20b();

function run(){
  if (drawnItems.getLayers().length == 1)
    drawnItems.clearLayers();
  
  g_selected_cluster = -1;
  time_changed(false);
  d3.select('#cluster_id').text('Cluster -1').style('color', '#000000');
  
  g_plot2 = $.plot("#timeseries2", [[], [], []], {});
}

function showTooltip(x, y, contents) {
  $("<div id='tooltip'>" + contents + "</div>").css({
    position: "absolute",
    display: "none",
    top: y + 5,
    left: x + 5,
    border: "1px solid #fdd",
    padding: "2px",
    "background-color": "#fee",
    opacity: 0.80
  }).appendTo("body").fadeIn(200);
}

function random_noise(){
  var r = Math.random() * (0.005 / 8) ;
  if (Math.random() > 0.5)
    r *= -1;
  return r;
}

function prev(){
  if (g_time){
    g_time -= 1000 * 60 * 60;
    
    g_pos -= 1;
    g_plot.unhighlight();
    g_plot.highlight(0, g_pos);
    
    time_changed(false);
  }
}

function  next(){
  if (g_time){
    g_time += 1000 * 60 * 60;
    
    g_pos += 1;
    g_plot.unhighlight();
    g_plot.highlight(0, g_pos);
    
    time_changed(false);
  }
}

function time_changed(fit){
  $('#loader').show();
  tmp_tweet_group = L.featureGroup();
  var min_time = g_time;
  var max_time = min_time + (60 * 60 * 1000);
  d3.select("#timestamp").text(moment.unix(g_time / 1000).format('LLLL'));
  
  // If we are tracking something using a polygon.
  if (drawnItems.getLayers().length == 1){
    spanish_twitter.getTweetsSVDD(session_id, g_dataset, min_time, max_time, polygon, polygon_extent, function(ret){ 
      polygon_word_cloud = ret.word_cloud;
      polygon_hash_cloud = ret.hash_cloud;
      data = ret['data']
      predicted = ret['predicted']
      data.forEach(function(d, i){
        var tweet = L.circle([d[0] + random_noise(), d[1] + random_noise()], radius(predicted[i]), {
          color: '#000000',
          fillColor: '#ff7f0e',
          weight: 2,
          fillOpacity: opacity(predicted[i]),
          time: d[2],
          screen_name: d[6],
          profile: d[7],
          inpolygon: d[9]
        });
        tweet.bindPopup(moment(d[2]).format('LLLL') + '<br>User: ' + d[6] + '<br>Tweet: ' + d[8] + '<br><br>' + '<div style="min-height: 40px"><a target="_blank" href="https://twitter.com/' + d[6] + '"><img width="48px" height="48px" style="float: left; padding-right: 5px" src="' + d[7] + '" /></a>' + d[5] + '</div>');
        tweet.on('click', function(e) {
           if(g_selected_cluster != this.options.cluster){
             g_selected_cluster = this.options.cluster;
             var color = colors(g_selected_cluster);
             if (g_selected_cluster == -1) color = '#000000';
             d3.select('#cluster_id').text('Cluster ' + g_selected_cluster).style('color', color);
             radio_changed();
           }
        });
        tmp_tweet_group.addLayer(tweet);
      });
      
      g_tweet_group.clearLayers();
      tmp_tweet_group.addTo(g_map);
      g_tweet_group = tmp_tweet_group;
      
      if(fit)
        g_map.fitBounds(g_tweet_group.getBounds());
      
      radio_changed();
      $('#loader').hide();
    });
  }
  
  else{
    spanish_twitter.getTweetsFor(g_dataset, min_time, max_time, $('#param_algorithm').val(), $('#param_eps').val(), $('#param_min_points').val(), function(ret){
      TMP = ret;
      g_tweetClouds = ret['tweetClouds'];
      g_hashClouds = ret['hashClouds'];
      g_selected_cluster = -1;
      data = ret['data']
      data.forEach(function(d){
        var color = colors(d[9]);
        if (d[10] < 0) color = '#000000';
        var tweet = L.circle([d[0] + random_noise(), d[1] + random_noise()], 50, {
          color: '#000000',
          fillColor: color,
          weight: 2,
          fillOpacity: 0.8,
          cluster: d[9],
          screen_name: d[6],
          time: d[2],
          profile: d[7]
        });
        tweet.bindPopup(moment(d[2]).format('LLLL') + '<br>User: ' + d[6] + '<br>Tweet: ' + d[8] + '<br>Cluster: ' + d[9] +'<br><br>' + '<div style="min-height: 40px"><a target="_blank" href="https://twitter.com/' + d[6] + '"><img width="48px" height="48px" style="float: left; padding-right: 5px" src="' + d[7] + '" /></a>' + d[3].replace('[pic]', '<a target="_blank" href="' + d[9] + '" >[pic]</a>') + '</div>');
        tweet.on('click', function(e) {
           if(g_selected_cluster != this.options.cluster){
             g_selected_cluster = this.options.cluster;
             var color = colors(g_selected_cluster);
             if (g_selected_cluster == -1) color = '#000000';
             d3.select('#cluster_id').text('Cluster ' + g_selected_cluster).style('color', color);
             radio_changed();
           }
        });
        tmp_tweet_group.addLayer(tweet);
      });
      
      g_tweet_group.clearLayers();
      tmp_tweet_group.addTo(g_map);
      g_tweet_group = tmp_tweet_group;
      
      if(fit)
        g_map.fitBounds(g_tweet_group.getBounds());
      
      radio_changed();
      $('#loader').hide();
    });
  }
}

function dataset_changed(){
  drawnItems.clearLayers();
  g_selected_cluster = -1;
  d3.select('#cluster_id').text('Cluster ' + g_selected_cluster).style('color', '#000000');
  g_tweetClouds = null;
  g_hashClouds = null;
  
  g_tweet_group.clearLayers();
  var value = $('#select_dataset').val();
  d3.json('data/' + value + '.json', function(data){
    $("#tooltip").remove();
    g_dataset = value;
    g_previous_point = null;
    g_plot = $.plot("#timeseries", [data.timeseries], {})
    var panRange = [0, parseInt(g_plot.getAxes().yaxis.ticks[g_plot.getAxes().yaxis.ticks.length-1].label)];
    g_plot = $.plot("#timeseries", [data.timeseries], {
      series: {
        lines: {
          show: true
        },
        shadowSize: 0,
        color: '#ff7f0e'
      },
      grid: {
        hoverable: true,
        clickable: true
      },
      xaxis: {
        mode: 'time',
        panRange: [data.min_time, data.max_time]
      },
      yaxis: {
        zoomRange: [1, 1],
        panRange: panRange
      },
      zoom: {
        interactive: true
      },
      pan: {
        interactive: true
      },
    });
    
    g_plot2 = $.plot("#timeseries2", [[], [], []], {});
    
    index = parseInt(g_plot.getData()[0].data.length / 2);
    g_time = g_plot.getData()[0].data[index][0];
    time_changed(true);
    g_plot.highlight(0, index);
  });
}

function word_cloud(data){
  if(data){
    var word_scale = d3.scale.linear()
      .domain(d3.extent(data.map(function(x){return x.size})))
      .range([10, 70]);
    
    d3.layout.cloud()
      .size([500, 500])
      .words(data)
      .rotate(function() { return 0; })
      .font("Arial")
      .fontSize(function(d) { return word_scale(d.size); })
      .on("end", draw)
      .start();
  }
}

function draw(words) {
  var text = vis.selectAll("text", function(d) { return d.text.toLowerCase(); })
    .data(words);
  
  text.transition()
    .duration(1000)
    .attr("transform", function(d) { return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")"; })
    .style("font-size", function(d) { return d.size + "px"; });
  
  text.enter().append("text")
    .attr("text-anchor", "middle")
    .attr("transform", function(d) { return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")"; })
    .style("font-size", function(d) { return d.size + "px"; })
    .style("opacity", 1e-6)
    .transition()
    .duration(1000)
    .style("opacity", 1);
  
  text.style("font-family", "Arial")
    .style("fill", function(d, i) { return fill(i); })
    .style('user-select', 'none')
    .style('cursor', 'pointer')
    .text(function(d) { return d.text; });
  
  var exitGroup = background.append("g")
    .attr("transform", vis.attr("transform"));
  
  var exitGroupNode = exitGroup.node();
  text.exit().each(function() {
    exitGroupNode.appendChild(this);
  });

  exitGroup.transition()
    .duration(1000)
    .style("opacity", 1e-6)
    .remove();
  
  vis.transition()
    .delay(1000)
    .duration(750)
    .attr("transform", "translate(" + [500 >> 1, 500 >> 1] + ")scale(" + 1 + ")");
}

function radio_changed(){
    d3.select("#word_cloud").selectAll("a").remove();
    if($("#radio_tweets")[0].checked){
      svg.style('display', 'block');
      d3.select("#word_cloud").style('overflow-y', 'hidden');
      if (g_selected_cluster == null){
        word_cloud(polygon_word_cloud);
      }
      else
        word_cloud(g_tweetClouds[g_selected_cluster]);
    }
    else if($("#radio_users")[0].checked){
      svg.style('display', 'none');
      d3.select("#word_cloud").style('overflow-y', 'scroll');
      var layers = g_tweet_group.getLayers();
      layers.sort(function(a,b){return a.options.time-b.options.time});
      if (g_selected_cluster == null){
        layers.forEach(function(layer){
          if (layer.options.inpolygon){
            $('#word_cloud').append('<a target="_blank" href="https://twitter.com/' + layer.options.screen_name + '"><img style="padding: 0px;" width="48px" height="48px" src="' + layer.options.profile + '" /></a>'); 
          }
        });
      }
      else{
        layers.forEach(function(layer){
          if (layer.options.cluster == g_selected_cluster){
            $('#word_cloud').append('<a target="_blank" href="https://twitter.com/' + layer.options.screen_name + '"><img style="padding: 0px;" width="48px" height="48px" src="' + layer.options.profile + '" /></a>'); 
          }
        });
      }
    }
    else if($("#radio_hashtags")[0].checked){
      svg.style('display', 'block');
      d3.select("#word_cloud").style('overflow-y', 'hidden');
      if (g_selected_cluster == null){
        word_cloud(polygon_hash_cloud);
      }
      else{
        word_cloud(g_hashClouds[g_selected_cluster]);
      }
    }
}

$('#loader').hide();
  
$("#timeseries").bind("plothover", function (event, pos, item) {
  if (item) {
    if (g_previous_point != item.dataIndex) {
      g_previous_point = item.dataIndex;
      $("#tooltip").remove();
      var x = item.datapoint[0];
      showTooltip(item.pageX, item.pageY, moment.unix(x / 1000).format('LLLL'));
    }
  }
});

$("#timeseries2").bind("plothover", function (event, pos, item) {
  if (item) {
    if (g_previous_point != item.dataIndex) {
      g_previous_point = item.dataIndex;
      $("#tooltip").remove();
      var x = item.datapoint[0];
      showTooltip(item.pageX, item.pageY, moment.unix(x / 1000).format('LLLL'));
    }
  }
});

$("#timeseries").bind("plotclick", function (event, pos, item) {
  if(g_previous_point)
    g_plot.unhighlight();
  if (item){
    g_pos = item.dataIndex;
    g_plot.highlight(item.series, item.datapoint);
    g_item = item;
    g_time = item.datapoint[0];
    time_changed(false);
  }
});

$("#timeseries2").bind("plotclick", function (event, pos, item) {
  if(g_previous_point)
    g_plot.unhighlight();
  if (item){
    g_plot.highlight(item.series, item.datapoint);
    g_item = item;
    g_time = item.datapoint[0];
    time_changed(false);
  }
});


$("#radio_tweets, #radio_hashtags, #radio_users").change(radio_changed);

g_map = new L.Map('map', {attributionControl: false}).setView([40.195, -3.757], 6);
var baseLayers = [
  'OpenStreetMap.BlackAndWhite',
  'OpenStreetMap.Mapnik',
  'Stamen.Toner',
];
L.control.layers.provided(baseLayers).addTo(g_map);

var drawnItems = new L.FeatureGroup();
g_map.addLayer(drawnItems);

var drawControl = new L.Control.Draw({
  draw: {
    position: 'topleft',
    polygon: {
        title: 'Draw a polygon',
        allowIntersection: false,
        drawError: {
            color: '#ff0000',
            timeout: 1000
        },
        shapeOptions: {
            color: '#ff0000',
            clickable: false
        }
    },
    rectangle: {
        title: 'Draw a rectangle',
        allowIntersection: false,
        drawError: {
            color: '#ff0000',
            timeout: 1000,
        },
        shapeOptions: {
            color: '#ff0000',
            fillOpacity: 0.1,
            opacity: 0.3,
            clickable: false
        }
    },
    circle: false,
    marker: false,
    polyline: false,
  },
  edit: {
      featureGroup: drawnItems,
      edit: false,
      remove: true
  }
});
g_map.addControl(drawControl);

function draw_timeline2(show_series){
  if (timeline2 != undefined){
    var series = [];
    series.push({data: timeline2.raw, lines:{show: show_series[0], fill: false, opacity: 0.1}, color: '#8c564b'});
    series.push({data: timeline2.routine,  lines:{show: show_series[1], fill: false}, color: '#1f77b4'});
    series.push({data: timeline2.events,  lines:{show: show_series[2], fill: true}, color: '#d62728'});
    
    g_plot2 = $.plot("#timeseries2", series, { });
    var panRange = [0, parseInt(g_plot2.getAxes().yaxis.ticks[g_plot2.getAxes().yaxis.ticks.length-1].label)];
    g_plot2 = $.plot("#timeseries2", series,{
      series: {
        lines: {
          show: true
        },
        shadowSize: 0
      },
      grid: {
        hoverable: true,
        clickable: true
      },
      xaxis: {
        mode: 'time',
        panRange: [timeline2.min_time, timeline2.max_time]
      },
      yaxis: {
        zoomRange: [1, 1],
        panRange: panRange
      },
      zoom: {
        interactive: true
      },
      pan: {
        interactive: true
      },
    });
  }
}

$('#series').on('click', 'li', function(d){
  topic_index = $(this).index();
  show_series[topic_index] = !show_series[topic_index];

  var elem = $('#series_' + (topic_index + 1));
  if (show_series[topic_index])
    elem.css("text-decoration", 'none');

  else
    elem.css("text-decoration", 'line-through');
  draw_timeline2(show_series);
});

g_map.on('draw:deleted', function (e) {
  if (drawnItems.getLayers().length == 1){
    drawnItems.clearLayers();
    g_selected_cluster = -1;
    time_changed(false);
    d3.select('#cluster_id').text('Cluster -1').style('color', '#000000');
    g_plot2 = $.plot("#timeseries2", [[], [], []], {});
  }
});

g_map.on('draw:created', function (e) {
  if(g_time == null){
    alert("A moment in time must be selected first!");
  }
  else{
    d3.select('#cluster_id').text('Polygon').style('color', '#ff7f0e');
    g_selected_cluster = null;
    d3.select("#word_cloud").selectAll("img").remove();
    
    $('#loader').show();
    drawnItems.clearLayers();
    
    type = e.layerType;
    layer = e.layer;
    
    drawnItems.addLayer(layer);
    
    wkt.fromObject(layer);
    polygon = wkt.write();
    
    spanish_twitter.get_tweets_for_polygon(session_id, g_dataset, polygon, g_time, function(data){
      polygon_extent = d3.extent(data.word_cloud.map(function(x){return x.size}));
      
      radius = d3.scale.linear()
         .range([20, 80])
         .domain(d3.extent(data.predicted))
      
      opacity = d3.scale.linear()
         .range([0.0, 0.8])
         .domain(d3.extent(data.predicted))
      
      timeline2 = data;
      draw_timeline2(show_series);
      
      time_changed(false);
      $('#loader').hide();
    })
  }
});

spanish_twitter.get_session_id(function(ret){
  session_id = ret;
  dataset_changed();
});

