import json
from traffic_processor import TrafficProcessor
import unittest

class TestTrafficProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TrafficProcessor()

    def test_sample_processing(self):
        # This matches the structure of the sample provided by the user (corrected syntax)
        sample_json = """
        {
          "01-08-2025": {
            "-0WaCEPkqzD7SO4TLVzM": {
              "date": "01-08-2025",
              "reason": "OpenRouter error: Payment Required",
              "status": "vehicles moving slowly",
              "suggestion": "Check traffic lights",
              "time": "20:46",
              "timestamp": 1754061402.536,
              "vehicle_count": 1
            },
            "-0WaCEPkqzD7SO4TLVzN": {
              "date": "01-08-2025",
              "reason": "Signal Issue",
              "status": "Signal Delay",
              "suggestion": "Optimize timing",
              "time": "20:50",
              "timestamp": 1754061642.536,
              "vehicle_count": 5
            }
          }
        }
        """
        processor = TrafficProcessor()
        data = processor.get_data_from_json(sample_json)
        result = processor.process_data(data)

        # Check Keys
        self.assertIn('vehicleCount', result)
        self.assertIn('time', result)
        self.assertIn('congestion', result)
        self.assertIn('report', result)
        self.assertIn('graph', result)  # Added by us for the app

        # Check Values
        self.assertEqual(result['vehicleCount'], 6)
        
        # We expect 2 items: "OpenRouter error: Payment Required" and "Signal Issue"
        names = [c['name'] for c in result['congestion']]
        self.assertIn("OpenRouter error: Payment Required", names)
        self.assertIn("Signal Issue", names)

        # Check percentages (each 50% since count is 1 for each category logic)
        # Wait, my logic counts occurrences of the category. 
        # Here we have 2 records. 1 "vehicles moving slowly", 1 "Signal Delay".
        # So 50/50 split.
        for c in result['congestion']:
            self.assertEqual(c['percentage'], 50.0)

        # Check daily_summary
        self.assertIn('daily_summary', result)
        summary = result['daily_summary']
        self.assertEqual(len(summary), 1) # One day in sample
        self.assertEqual(summary[0]['date'], "01-08-2025")
        self.assertEqual(summary[0]['totalEvents'], 2)
        self.assertEqual(summary[0]['peakVehicleCount'], 5)

        # Check detailed_history
        self.assertIn('detailed_history', result)
        history = result['detailed_history']
        self.assertEqual(len(history), 2)
        # Check first item (newest first)
        self.assertEqual(history[0]['reason'], "Signal Issue") # 20:47 > 20:46

        print("\nTest Output Result:\n", json.dumps(result, indent=2))

    def test_few_days_processing(self):
        """Test that summary is generated even if we have fewer than 10 days"""
        multi_day_sample = {
            "01-08-2025": {
                "id1": { "vehicle_count": 5, "reason": "R1", "timestamp": 1000 }
            },
            "02-08-2025": {
                "id2": { "vehicle_count": 3, "reason": "R2", "timestamp": 2000 }
            }
        }
        processed = self.processor.process_data(multi_day_sample)
        daily = processed.get('daily_summary', [])
        
        # Should have 2 entries (using available days)
        self.assertEqual(len(daily), 2)
        dates = sorted([d['date'] for d in daily])
        self.assertEqual(dates, ["01-08-2025", "02-08-2025"])
        print("\n[Test] Verified handling of few days:", dates)

if __name__ == '__main__':
    unittest.main()
