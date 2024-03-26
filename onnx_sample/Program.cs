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

namespace Onnx_sample
{
    struct Box
    {
        public int x1, x2, y1, y2;
        public float score;
    }
    class Program
    {
        const int INP_SIZE = 640;
        static float scale_x, scale_y;
        static int orig_w, orig_h;
        public static void Main(string[] args)
        {
            string modelFilePath = "./mot17-01-frcnn.onnx";
            using var session = new InferenceSession(modelFilePath);

            // Read Image
            string imageFilePath = "./images/000001.jpg";
            using Image<Rgb24> image = Image.Load<Rgb24>(imageFilePath);
            var image_orig = image.Clone();


            // Resize image
            scale_x = (float)image.Width / (float)INP_SIZE;
            scale_y = (float)image.Height / (float)INP_SIZE;
            image.Mutate(x => x.Resize(640, 640));

            // Prepare input tensor
            var images = new List<Image<Rgb24>>() { image };
            Tensor<float> input = ConvertImageToTensor(ref images, [1, 3, 640, 640]);

            // Run model
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", input)
            };


            var result = session.Run(inputs);
            Tensor<float> resultTensor = result[0].AsTensor<float>();
            var final_boxes = Postprocess(ref resultTensor);
            foreach (var box in final_boxes)
            {
                int center_x = (box.x1 + box.x2) / 2;
                int center_y = (box.y1 + box.y2) / 2;
                int width = Math.Abs(box.x2 - box.x1);
                int height = Math.Abs(box.y2 - box.y1);
                Rectangle rectangle = new Rectangle(x: box.x1, y: box.y1, width: width, height: height);
                var pen = Pens.Solid(Color.Red, 2);
                image_orig.Mutate(x=>x.Draw(pen, rectangle));
            }
            image_orig.Save("./out.jpg");


            
            Console.WriteLine("done");
        }

        public static Tensor<float> ConvertImageToTensor(ref List<Image<Rgb24>> images, int[] inputDimension)
        {
            inputDimension[0] = images.Count;

            Tensor<float> input = new DenseTensor<float>(inputDimension);

            for (var i = 0; i < images.Count; i++)
            {
                var image = images[i];
                image.ProcessPixelRows(accessor =>
                {
                    for (var y = 0; y < image.Height; y++)
                    {
                        Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                        for (var x = 0; x < image.Width; x++)
                        {
                            input[i, 0, y, x] = (float)pixelSpan[x].R /255.0f;
                            input[i, 1, y, x] = (float)pixelSpan[x].G/255.0f;
                            input[i, 2, y, x] = (float)pixelSpan[x].B/255.0f;
                        }
                    }
                }
                );
               
            }
            return input;
        }

        public static List<Box> Postprocess(ref Tensor<float> outTensor)
        {
            var correct = new List<Box>();
            for(int i =0; i < outTensor.Dimensions[2];  i++)
            {
                if (outTensor[0, 4, i] >0.3)
                {
                    var box = new Box();
                    box.x1 = (int)((outTensor[0, 0, i] - outTensor[0, 2, i] / 2) * scale_x);
                    box.x2 = (int)((outTensor[0, 0, i] + outTensor[0, 2, i] / 2) * scale_x);
                    box.y1 = (int)((outTensor[0, 1, i] - outTensor[0, 3, i] / 2) * scale_y);
                    box.y2 = (int)((outTensor[0, 1, i] + outTensor[0, 3, i] / 2) * scale_y);
                    box.score = outTensor[0, 4, i];
                    correct.Add(box);
                }
            }
            return NMS(ref correct);
        }

        public static List<Box>NMS(ref List<Box> correct, float thr=0.5f)
        {
            if (correct.Count == 0)
                return new List<Box>();
            correct = correct.OrderBy(x => x.score).ToList();
            var final_boxes = new List<Box>();
            while(correct.Count > 0)
            {
                var current_box = correct[correct.Count - 1];
                correct.RemoveAt(correct.Count - 1);
                final_boxes.Add(current_box);
                
                correct = correct.Where(box  => IOU(current_box, box) <= thr).ToList();
            }
            return final_boxes;
        }
        
        public static float IOU(Box box1, Box box2)
        {
            float x_overlap = Math.Max(0,  Math.Min(box1.x2, box2.x2)-Math.Max(box1.x1, box2.x1)); 
            float y_overlap = Math.Max(0,  Math.Min(box1.y2, box2.y2)-Math.Max(box1.y1, box2.y1)); 
            float overlap_area = x_overlap * y_overlap;
            float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
            float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
            float total_area = area1 + area2 - overlap_area;
            return overlap_area/total_area;
        }
    }
}
