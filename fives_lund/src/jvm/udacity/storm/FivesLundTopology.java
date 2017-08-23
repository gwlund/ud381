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

import udacity.storm.FivesLundPLPPythonBolt;

import com.lambdaworks.redis.RedisClient;
import com.lambdaworks.redis.RedisConnection;

import java.util.Map;

public class FivesLundTopology {

	public static void main(String[] args) throws Exception {
		// create the topology
		String topologyName = "ipi_algo";
		
		TopologyBuilder builder = new TopologyBuilder();

		// attach the spout to the topology - parallelism of 1
		//builder.setSpout("spout", new FivesLundPLPSpout());
		builder.setSpout("test_spout", new FivesLundTestSpout());

		// attach the bolt to the topology - parallelism of 1
		//builder.setBolt("bolt", new FivesLundPLPPythonBolt()).shuffleGrouping("spout");
		builder.setBolt("test_bolt", new FivesLundTestBolt()).shuffleGrouping("test_spout", "located").shuffleGrouping("test_spout", "zs").shuffleGrouping("test_spout", "xs");

		// create the default config object
		Config conf = new Config();

		// set the config in debugging mode
		conf.setDebug(true);

		if (args != null && args.length > 0) {

			// run it in a live cluster
			System.out.println("Running in a Live Cluster");

			// set the number of workers for running all spout and bolt tasks
			conf.setNumWorkers(3);

			// create the topology and submit with config
			StormSubmitter.submitTopology(args[0], conf, builder.createTopology());

		} else {

			// run it in a simulated local cluster
			System.out.println("Running in a Simulated Cluster");

			// create the local cluster instance
			LocalCluster cluster = new LocalCluster();

			// submit the topology to the local cluster
			cluster.submitTopology(topologyName, conf, builder.createTopology());

			// let the topology run for 30 seconds. note topologies never terminate!
			Thread.sleep(30000);

			// kill the topology
			cluster.killTopology(topologyName);

			// we are done, so shutdown the local cluster
			cluster.shutdown();
		}
	}

}
