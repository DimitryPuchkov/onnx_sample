using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.Processing;

namespace onnx_sample
{
    [MemoryDiagnoser]
    public class TrackerBenchmarker
    {

        static string[] imagesPaths = { "./images/000001.jpg", "./images/000002.jpg", "./images/000003.jpg" };
        static DetectionWrapper detector = new DetectionWrapper();
        static Sort sort = new Sort(15);
        static Image<Rgb24> image = Image.Load<Rgb24>(imagesPaths[0]);
        static InferenceSession session16 = new InferenceSession("./model_fp16.onnx");
        static List<NamedOnnxValue> inps16 = Preprocess16(image);

        static List<Image<Rgb24>> images = getImageBatch();
        static List<NamedOnnxValue> inpsBatch = PreprocessBatch(images);
        static InferenceSession sessionBatch = new InferenceSession("./mot17_dim.onnx");


        static InferenceSession session32 = new InferenceSession("./mot17-01-frcnn.onnx");
        static List<NamedOnnxValue> inps32 = Preprocess32(image);

        private static List<NamedOnnxValue> Preprocess16(Image<Rgb24> image)
        {
            image.Mutate(x => x.Resize(AppConfig.ModelInputSize, AppConfig.ModelInputSize));
            int[] inputDimension = [1, 3, 640, 640];

            Tensor<Float16> input = new DenseTensor<Float16>(inputDimension);

            image.ProcessPixelRows(accessor =>
            {
                for (var y = 0; y < image.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (var x = 0; x < image.Width; x++)
                    {
                        input[0, 0, y, x] = (Float16)((float)pixelSpan[x].R / 255.0f);
                        input[0, 1, y, x] = (Float16)((float)pixelSpan[x].G / 255.0f);
                        input[0, 2, y, x] = (Float16)((float)pixelSpan[x].B / 255.0f);
                    }
                }
            });

            return new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", input) };
        }

        private static List<NamedOnnxValue> Preprocess32(Image<Rgb24> image)
        {
            image.Mutate(x => x.Resize(AppConfig.ModelInputSize, AppConfig.ModelInputSize));
            int[] inputDimension = [1, 3, 640, 640];

            Tensor<float> input = new DenseTensor<float>(inputDimension);

            image.ProcessPixelRows(accessor =>
            {
                for (var y = 0; y < image.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (var x = 0; x < image.Width; x++)
                    {
                        input[0, 0, y, x] = ((float)pixelSpan[x].R / 255.0f);
                        input[0, 1, y, x] = ((float)pixelSpan[x].G / 255.0f);
                        input[0, 2, y, x] = ((float)pixelSpan[x].B / 255.0f);
                    }
                }
            });

            return new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", input) };
        }

        private static List<NamedOnnxValue> Preprocess32Mod(Image<Rgb24> image)
        {
            image.Mutate(x => x.Resize(AppConfig.ModelInputSize, AppConfig.ModelInputSize));
            int[] inputDimension = [1, 3, 640, 640];

            Tensor<float> input = new DenseTensor<float>(inputDimension);

            //image.ProcessPixelRows(accessor =>
            //{
            //    for (var y = 0; y < image.Height; y++)
            //    {
            //        Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
            //        for (var x = 0; x < image.Width; x++)
            //        {
            //            input[0, 0, y, x] = ((float)pixelSpan[x].R / 255.0f);
            //            input[0, 1, y, x] = ((float)pixelSpan[x].G / 255.0f);
            //            input[0, 2, y, x] = ((float)pixelSpan[x].B / 255.0f);
            //        }
            //    }
            //});

            for(var i =  0; i < 640; i++)
                for(var j = 0; j < 640; j++)
                {
                    input[0, 0, i, j] = (float)i;
                    input[0, 1, i, j] = (float)j;
                    input[0, 2, i, j] = (float)(i+j);
                }


            return new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", input) };
        }

        private static List<Image<Rgb24>> getImageBatch()
        {
            var imagesBatch = new List<Image<Rgb24>>();
            for (int i = 0; i < 30; i++)
            {
                imagesBatch.Add(Image.Load<Rgb24>(imagesPaths[0]));
            }
            return imagesBatch;
        }


        private static List<NamedOnnxValue> PreprocessBatch(List<Image<Rgb24>> images)
        {
            int[] inputDimension = [images.Count, 3, 640, 640];
            Tensor<float> input = new DenseTensor<float>(inputDimension);

            for (int i = 0; i < inputDimension[0]; i++)
            {
                images[i].Mutate(x => x.Resize(AppConfig.ModelInputSize, AppConfig.ModelInputSize));
                images[0].ProcessPixelRows(accessor =>
                {
                    for (var y = 0; y < images[i].Height; y++)
                    {
                        Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                        for (var x = 0; x < images[i].Width; x++)
                        {
                            input[i, 0, y, x] = (float)pixelSpan[x].R / 255.0f;
                            input[i, 1, y, x] = (float)pixelSpan[x].G / 255.0f;
                            input[i, 2, y, x] = (float)pixelSpan[x].B / 255.0f;
                        }
                    }
                });
            }
            return new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", input) };
        }

        [Benchmark]
        public void RunPreprocess32()
        {
            var res = Preprocess32(image);

        }

        //[Benchmark]
        //public void RunPreprocess16()
        //{
        //    var res = Preprocess16(image);

        //}

        [Benchmark]
        public void RunPreprocess32Mod()
        {
            var res = Preprocess32Mod(image);

        }


    }

}
