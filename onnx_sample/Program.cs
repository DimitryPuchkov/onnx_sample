using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using onnx_sample;
using MathNet.Filtering.Kalman;
using MathNet.Numerics.LinearAlgebra;
using LinearAssignment;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Configs;
using System.Drawing;
using NumSharp;
using MathNet.Numerics;
using System.Drawing.Imaging;

namespace Onnx_sample
{

    class Program
    {
        public static void Main(string[] args)
        {

            //string[] imagesPaths = { "./images/000001.jpg", "./images/000002.jpg", "./images/000003.jpg" };
            //var detector = new DetectionWrapper();
            ////var sort = new Sort(15);
            //SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24> imageSL = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgb24>(imagesPaths[0]);
            //var scale_x = (double)imageSL.Width / (double)AppConfig.ModelInputSize;
            //var scale_y = (double)imageSL.Height / (double)AppConfig.ModelInputSize;
            //var images = new List<Image<Rgb24>>();
            //for (int i = 0; i < 30; i++)
            //{
            //    images.Add(Image.Load<Rgb24>(imagesPaths[0]));
            //}
            //var detsBatch = detector.DetectBatch(images);
            //var watch = new System.Diagnostics.Stopwatch();
            //watch.Start();
            //for (int i = 0; i < 30; i++)
            //{
            //    var dets = detector.Detect(image);
            //    var finalBoxes = sort.Update(dets);
            //}
            //watch.Stop();

            //Console.WriteLine($"Execution Time: {watch.ElapsedMilliseconds} ms");

            //foreach (string imagePath in imagesPaths)
            //{
            //    using Image<Rgb24> image = Image.Load<Rgb24>(imagePath);
            //    var dets = detector.Detect(image);
            //    var finalBoxes = sort.Update(dets);
            //    Console.WriteLine(finalBoxes);
            //}

            //Bitmap img = new Bitmap(imagesPaths[0]);
            //int[] inputDimension = [1, 3, 640, 640];
            //img = new Bitmap(img, new Size(640, 640));
            //img = img.Clone(new Rectangle(0, 0, img.Width, img.Height), PixelFormat.Format24bppRgb);
            //var ndarray = img.ToNDArray(flat: false, copy: false);
            //ndarray = ndarray.astype(typeof(float));
            //ndarray /= 255;
            //ndarray = ndarray.transpose(new int[] { 0, 3, 1, 2 });
            //Tensor<float> inputTensor = new DenseTensor<float>(memory: ndarray.ToArray<float>(), inputDimension);
            //var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", inputTensor) };
            //var npout = detector.session.Run(input);
            //var npres = detector.Postprocess32(npout[0].AsTensor<float>(), 1.0, 1.0);

            //var commonres = detector.Detect(imageSL);


            var config = ManualConfig.Create(DefaultConfig.Instance).WithOptions(ConfigOptions.DisableOptimizationsValidator);
            BenchmarkRunner.Run<ExperimentBenchhmarker>(config);
            //BenchmarkRunner.Run<TrackerBenchmarker>(config);
        }


        //public static void DetectionSample()
        //{
        //    var detector = new DetectionWrapper();


        //    // Read Image
        //    string imageFilePath = "./imagesPaths/000001.jpg";
        //    using Image<Rgb24> image = Image.Load<Rgb24>(imageFilePath);
        //    var image_orig = image.Clone();

        //    var final_boxes = detector.Detect(image);
        //    DrawBoxes(image, final_boxes);
        //}

        //public static void DrawBoxes(Image<Rgb24> image, List<Box> boxes, string imagePath = "out.jpg")
        //{
        //    foreach (var box in boxes)
        //    {
        //        int center_x = (box.x1 + box.x2) / 2;
        //        int center_y = (box.y1 + box.y2) / 2;
        //        int width = Math.Abs(box.x2 - box.x1);
        //        int height = Math.Abs(box.y2 - box.y1);
        //        Rectangle rectangle = new Rectangle(x: box.x1, y: box.y1, width: width, height: height);
        //        var pen = Pens.Solid(Color.Red, 2);
        //        image.Mutate(x => x.Draw(pen, rectangle));
        //    }
        //    image.Save(imagePath);
        //}
    }
}
