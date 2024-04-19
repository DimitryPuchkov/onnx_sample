using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Configuration;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp.PixelFormats;
using Microsoft.ML.OnnxRuntime.Tensors;
using static System.Net.Mime.MediaTypeNames;
using SixLabors.ImageSharp.Processing;

namespace onnx_sample
{

    public struct Box
    {
        public int x1, x2, y1, y2, object_id;
        public float score;

        public Box()
        {
            object_id = -1;
        }
    }

    public class DetectionWrapper
    {
        // Обертка над моделью детекции, включает в себя сессию onnxruntime и различные функии для препроцесса и постпроцесса вида:
        // Preprocess[16|32|Batch] и Postprocess[16|32|Batch] для запуска квантизованной модели (fp16), модели в обычном режиме (fp32) и батчем по 30 кадровЫ
        // Главные функции детектора Dectct и DetectBatch
        private InferenceSession session = new InferenceSession(AppConfig.ModelPath);

        public List<Box> Detect(Image<Rgb24> image)
        {
            image = image.Clone();
            double scale_x, scale_y;
            scale_x = (double)image.Width / (double)AppConfig.ModelInputSize;
            scale_y = (double)image.Height / (double)AppConfig.ModelInputSize;

            var modelInput = Preprocess32(image);
            var modelOutput = session.Run(modelInput);

            return Postprocess32(modelOutput[0].AsTensor<float>(), scale_x, scale_y);

        }
        public List<List<Box>> DetectBatch(List<Image<Rgb24>> images)
        {
            for (int i = 0; i < images.Count; i++) images[i] = images[i].Clone();
            double scale_x, scale_y;
            scale_x = (double)images[0].Width / (double)AppConfig.ModelInputSize;
            scale_y = (double)images[0].Height / (double)AppConfig.ModelInputSize;

            var modelInput = PreprocessBatch(images);
            var modelOutput = session.Run(modelInput);

            return PostprocessBatch(modelOutput[0].AsTensor<float>(), scale_x, scale_y);

        }

        private List<NamedOnnxValue> Preprocess16(Image<Rgb24> image)
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
                        input[0, 0, y, x] = (Float16) ((float)pixelSpan[x].R / 255.0f);
                        input[0, 1, y, x] = (Float16)((float)pixelSpan[x].G / 255.0f);
                        input[0, 2, y, x] = (Float16)((float)pixelSpan[x].B / 255.0f);
                    }
                }
            });

            return new List<NamedOnnxValue>{NamedOnnxValue.CreateFromTensor("images", input)};
        }

        private List<NamedOnnxValue> PreprocessBatch(List<Image<Rgb24>> images)
        {
            int[] inputDimension = [images.Count, 3, 640, 640];
            Tensor<float> input = new DenseTensor<float>(inputDimension);

            for(int i = 0; i < inputDimension[0]; i++)
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


        private List<Box> Postprocess16(Tensor<Float16> outTensor, double scale_x, double scale_y)
        {
            var correct = new List<Box>();
            for (int i = 0; i < outTensor.Dimensions[2]; i++)
            {
                if ((float)outTensor[0, 4, i] > 0.3)
                {
                    var box = new Box();
                    box.x1 = (int)( ((float)outTensor[0, 0, i] - (float) outTensor[0, 2, i] / 2) * scale_x);
                    box.x2 = (int)(((float)outTensor[0, 0, i] + (float)outTensor[0, 2, i] / 2) * scale_x);
                    box.y1 = (int)(((float)outTensor[0, 1, i] - (float)outTensor[0, 3, i] / 2) * scale_y);
                    box.y2 = (int)(((float)outTensor[0, 1, i] + (float)outTensor[0, 3, i] / 2) * scale_y);
                    box.score = (float)outTensor[0, 4, i];
                    correct.Add(box);
                }
            }
            return Utils.NMS(ref correct);
        }

        private List<List<Box>> PostprocessBatch(Tensor<float> outTensor, double scale_x, double scale_y)
        {
            var correctBatch = new List<List<Box>>();
            for (int i = 0; i < outTensor.Dimensions[0]; i++)
            {
                var correct = new List<Box>();
                for (int j = 0; j < outTensor.Dimensions[2]; j++)
                {
                    if ((float)outTensor[i, 4, j] > 0.3)
                    {
                        var box = new Box();
                        box.x1 = (int)(((float)outTensor[i, 0, j] - (float)outTensor[0, 2, j] / 2) * scale_x);
                        box.x2 = (int)(((float)outTensor[i, 0, j] + (float)outTensor[0, 2, j] / 2) * scale_x);
                        box.y1 = (int)(((float)outTensor[i, 1, j] - (float)outTensor[0, 3, j] / 2) * scale_y);
                        box.y2 = (int)(((float)outTensor[i, 1, j] + (float)outTensor[0, 3, j] / 2) * scale_y);
                        box.score = (float)outTensor[i, 4, j];
                        correct.Add(box);
                    }
                }
                correctBatch.Add(Utils.NMS(ref correct));
            }

            
            return correctBatch;
        }

        private List<NamedOnnxValue> Preprocess32(Image<Rgb24> image)
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
                        input[0, 0, y, x] = (float)pixelSpan[x].R / 255.0f;
                        input[0, 1, y, x] = (float)pixelSpan[x].G / 255.0f;
                        input[0, 2, y, x] = (float)pixelSpan[x].B / 255.0f;
                    }
                }
            });

            return new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", input) };
        }

        private List<Box> Postprocess32(Tensor<float> outTensor, double scale_x, double scale_y)
        {
            var correct = new List<Box>();
            for (int i = 0; i < outTensor.Dimensions[2]; i++)
            {
                if (outTensor[0, 4, i] > 0.3)
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
            return Utils.NMS(ref correct);
        }
    }

}
