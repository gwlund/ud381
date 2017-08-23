package udacity.storm;

import backtype.storm.Config;
import backtype.storm.LocalCluster;
import backtype.storm.StormSubmitter;
import backtype.storm.task.OutputCollector;
import backtype.storm.task.ShellBolt;
import backtype.storm.task.TopologyContext;
import backtype.storm.testing.TestWordSpout;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.TopologyBuilder;
import backtype.storm.topology.base.BaseRichBolt;
import backtype.storm.topology.IRichBolt;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Tuple;
import backtype.storm.tuple.Values;
import backtype.storm.utils.Utils;

import java.util.Map;

public class FivesLundPLPPythonBolt extends ShellBolt implements IRichBolt
{
    // To output tuples from this bolt to the next stage bolts, if any
    OutputCollector _collector;
    
    public FivesLundPLPPythonBolt()
    {
    	super("python", "fivesbolt1.py");
    }
    
    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer)
    {
      // tell storm the schema of the output tuple for this spout

      // tuple consists of a single column called 'results'
      declarer.declare(new Fields("results"));
    }
	
	@Override
	public Map<String, Object> getComponentConfiguration()
	{
		return null;
	}

}
