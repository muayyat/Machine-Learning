using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
namespace SR
{
    public class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        public static void Main(string[] args)
        {
            MLContext mLContext = new MLContext(seed: 0);
            var model = Train(mLContext, _trainDataPath);
            Evaluate(mLContext, model);
            TestSinglePrediction(mLContext, model);
        }
      
        public static ITransformer Train(MLContext mLContext, string dataPath)
        {
            IDataView dataView = mLContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');
            var pipeline = mLContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(mLContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripTime", "TripDistance", "PaymentTypeEncoded"))
                .Append(mLContext.Regression.Trainers.FastTree())
                ;
            var model = pipeline.Fit(dataView);
            return model;
            
        }
        private static void Evaluate(MLContext mLContext, ITransformer model)
        {
            IDataView dataView = mLContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(dataView);
            var metrics = mLContext.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine();
            Console.WriteLine($"*   &copy; Muayyat Billah  ");
            Console.WriteLine($"*   Model Evaluation     ");
            Console.WriteLine($"*------------------------");
            Console.WriteLine($"* RS Score: {metrics.RSquared:0.##}");
            Console.WriteLine($"* RMS Error: {metrics.RootMeanSquaredError:#.##}");

        }
        private static void TestSinglePrediction(MLContext mLContext, ITransformer model)
        {
            var predictionFunction = mLContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "CMT",
                RateCode = "1",
                PassengerCount = 12,
                TripTime = 1149,
                TripDistance = 1913.75f,
                PaymentType = "CSH",
                FareAmount = 0 // To predict. Actual/Observed = 15.5

            };
            var prediction = predictionFunction.Predict(taxiTripSample);
            Console.WriteLine($"***********************");
            Console.WriteLine($"Passengers: {taxiTripSample.PassengerCount}");
            Console.WriteLine($"Drive Time: {taxiTripSample.TripTime}");
            Console.WriteLine($"Predicted Price: {prediction.FareAmount: 0.####}, Actual Price:16.039");
            Console.WriteLine($"***********************");

        }
     ///   public static IWebHostBuilder CreateWebHostBuilder(string[] args) =>
       //     WebHost.CreateDefaultBuilder(args)
    //            .UseStartup<Startup>();
    }
}
