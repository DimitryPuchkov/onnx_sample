using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using onnx_sample;
using MathNet.Filtering.Kalman;
using MathNet.Numerics.LinearAlgebra;
using LinearAssignment;

namespace Onnx_sample
{

    class Program
    {
        public static void Main(string[] args)
        {







            Console.WriteLine("done");
        }

        public static void DetectionSample()
        {
            var detector = new DetectionWrapper();


            // Read Image
            string imageFilePath = "./images/000001.jpg";
            using Image<Rgb24> image = Image.Load<Rgb24>(imageFilePath);
            var image_orig = image.Clone();

            var final_boxes = detector.Detect(image);
            DrawBoxes(image, final_boxes);
        }

        public static void DrawBoxes(Image<Rgb24> image, List<Box> boxes, string imagePath = "out.jpg")
        {
            foreach (var box in boxes)
            {
                int center_x = (box.x1 + box.x2) / 2;
                int center_y = (box.y1 + box.y2) / 2;
                int width = Math.Abs(box.x2 - box.x1);
                int height = Math.Abs(box.y2 - box.y1);
                Rectangle rectangle = new Rectangle(x: box.x1, y: box.y1, width: width, height: height);
                var pen = Pens.Solid(Color.Red, 2);
                image.Mutate(x => x.Draw(pen, rectangle));
            }
            image.Save(imagePath);
        }
    }
}
