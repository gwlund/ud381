package udacity.storm;

import backtype.storm.spout.SpoutOutputCollector;
import backtype.storm.task.TopologyContext;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.base.BaseRichSpout;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Values;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import com.opencsv.CSVReader;

public class FivesLundPLPSpout extends BaseRichSpout
{
	private SpoutOutputCollector _collector;
	
	private CSVReader locatedReader;
	private CSVReader profileReader;
	private AtomicLong linesReadLocated;
	private AtomicLong linesReadProfile;
	private AtomicLong measurementId;
	
	private String locatedFileName = "/vagrant/fives_lund/src/resources/located_profiles.csv";
	private String profileFileName = "/vagrant/fives_lund/src/resources/zs.csv";
	
	public FivesLundPLPSpout()
	{
		
	}
	
	@Override
	public void open(Map conf, TopologyContext context, SpoutOutputCollector collector)
	{
		_collector = collector;
		//TODO: Get filename from config
		
		System.out.println("Opening PLP Files");
		
		locatedReader = openFile(locatedFileName, true);
		profileReader = openFile(profileFileName, false);
		
		measurementId = new AtomicLong(1);
		linesReadLocated = new AtomicLong(0);
		linesReadProfile = new AtomicLong(0);
	}
	
	@Override
	public void nextTuple()
	{
		
		String[] values;
		
		if (measurementId.get() == 1)
		{
			System.out.println("Outputing xs");
			values = readLine(profileReader, linesReadProfile);
			_collector.emit("xs", new Values(values));
			
		}
		
		values = readLine(profileReader, linesReadProfile);
		//TODO: Emit Z-values
		_collector.emit("zs", new Values(values));
		
		values = readLine(locatedReader, linesReadLocated);
		//TODO: Emit located-values
		_collector.emit("located", new Values(values));
		
		measurementId.incrementAndGet();
			
	}
	
	@Override
	public void ack(Object id) 
	{
		
	}

	@Override
	public void fail(Object id) 
	{
		System.err.println("Failed tuple with id "+id);
	}
	
	@Override
	public void declareOutputFields(OutputFieldsDeclarer declarer) 
	{
		System.out.println("Declaring PLP Fields");
		Fields profileFields = getProfileFields();
		Fields locatedFields = getLocatedFields();

		// Declare two streams
		declarer.declareStream("xs", profileFields);
		declarer.declareStream("zs", profileFields);
		declarer.declareStream("located", locatedFields);

	}
	  
	private Fields getLocatedFields() {
		try {
			System.out.println("Getting Located Field Names");
			CSVReader reader = new CSVReader(new FileReader(locatedFileName), ',');
			// read csv header to get field info
			String[] fields = reader.readNext();

			System.out.println("DECLARING OUTPUT FIELDS");

			for (String a : fields)
				System.out.println(a);

			Fields locFields = new Fields(Arrays.asList(fields));
			return locFields;

		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	private Fields getProfileFields() {
		try {
			
			System.out.println("Getting Profile Field Names");
			CSVReader reader = new CSVReader(new FileReader(profileFileName), ',');
			// read csv header to get field info
			String[] fieldNames = reader.readNext();

			// if there are no headers, just use field_index naming convention
			ArrayList<String> f = new ArrayList<String>(fieldNames.length);

			for (int i = 0; i < fieldNames.length; i++) {
				f.add("field_" + i);
			}

			Fields fields = new Fields(f);
			return fields;

		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public String[] readLine(CSVReader reader, AtomicLong linesRead) 
	  {
		  String[] line = null;
		    try {
		      line = reader.readNext();
		      if (line != null) {
		        long id = linesRead.incrementAndGet();
		        System.out.println(Arrays.toString(line));
		        //_collector.emit(new Values(line), id);
	      } else {
	        System.out.println("Finished reading file, " + linesRead.get() + " lines read");
	      }

	    } catch (Exception e) {
	      e.printStackTrace();
	    }
			return line;

	  }
		
	  public CSVReader openFile(String fileName, boolean includesHeaderRow) 
	  {

		CSVReader reader;
		try {
		  //reader = new BufferedReader(new FileReader(fileName));
		      reader = new CSVReader(new FileReader(fileName), ',');
		      // read and ignore the header if one exists
		      if (includesHeaderRow) reader.readNext();
		  // read and ignore the header if one exists
		  

		} catch (Exception e) {
		  throw new RuntimeException(e);
		}
		
		return reader;

	  }

} //end class
