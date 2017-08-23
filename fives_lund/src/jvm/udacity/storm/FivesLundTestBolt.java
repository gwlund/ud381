package udacity.storm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import backtype.storm.Config;
import backtype.storm.LocalCluster;
import backtype.storm.StormSubmitter;
import backtype.storm.task.OutputCollector;
import backtype.storm.task.TopologyContext;
import backtype.storm.testing.TestWordSpout;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.TopologyBuilder;
import backtype.storm.topology.base.BaseRichBolt;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Tuple;
import backtype.storm.tuple.Values;
import backtype.storm.utils.Utils;

public class FivesLundTestBolt extends BaseRichBolt
{
  // To output tuples from this bolt to the next stage bolts, if any
  OutputCollector _collector;
  Map<String, List<Integer>> xsMap = new HashMap<String, List<Integer>>();

  @Override
  public void prepare(
      Map                     map,
      TopologyContext         topologyContext,
      OutputCollector         collector)
  {
    // save the output collector for emiting tuples
    _collector = collector;

    
  }

  @Override
  public void execute(Tuple tuple)
  {
	  
	  String sourceComponent = tuple.getSourceComponent();
	  String streamId = tuple.getSourceStreamId();
	  
	  //System.out.println("Execute Test Bolt: Component: " + sourceComponent + " streamId: " + streamId);
	  if (streamId == "zs")
	  {
		  
	  } else if (streamId == "located")
	  {
		  
	  } else if (streamId == "xs")
	  {
		  //Example populating map
		  //TODO: Need to iterate through tuple and create list
		  xsMap.put(tuple.getString(0), new ArrayList<Integer>(Arrays.asList(0, 4, 8, 9, 12)));
		  
	  }
	  
	    // get the column word from tuple
	    String word = tuple.getString(0);

    // emit the word with exclamations
    _collector.emit(tuple, new Values(word));
  }

  @Override
  public void declareOutputFields(OutputFieldsDeclarer declarer)
  {
    // tell storm the schema of the output tuple for this spout

    // tuple consists of a single column called 'test-word'
    declarer.declare(new Fields("test-word"));
  }

}
