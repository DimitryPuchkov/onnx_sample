using LinearAssignment;
using MathNet.Numerics.LinearAlgebra.Complex;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace onnx_sample
{
    public class Sort
    {
        private int maxAge;
        private double iouThresh;
        private List<KalmanBoxTracker> trackers;
        public Sort(int maxAge, double iouThresh=0.3) 
        {   
            this.maxAge = maxAge;
            this.iouThresh = iouThresh;
            trackers = new List<KalmanBoxTracker>();
        }

        public List<Box> Update(List<Box> dets)
        {
            foreach (var tracker in trackers)
                tracker.Predict();

            var associateResult = associateDetectionsToTrackers(dets);
            var matches = associateResult.Item1;
            // Update matched trackers
            foreach(var match in matches)
                trackers[match[1]].Update(dets[match[0]]);

            // Create new trackers from unmathched detections
            foreach (Box box in associateResult.Item2)
                trackers.Add(new KalmanBoxTracker(box));

            var resultBoxes = new List<Box>();
            for(int i = trackers.Count - 1; i >= 0; i--)
            {
                var tracker = trackers[i];
                if (tracker.TimeScinceUpdate == 0)
                {
                    var box = tracker.GetState();
                    box.object_id = tracker.TrackerID;
                    resultBoxes.Add(box);
                }
                    
                if (tracker.TimeScinceUpdate > maxAge)
                    trackers.RemoveAt(i);
            }
            return resultBoxes;
        }

        private Tuple<List<List<int>>, List<Box>> associateDetectionsToTrackers(List<Box> detections) 
        {
            if (trackers.Count ==0)
                return Tuple.Create(new List<List<int>>(), detections);

            List<Box> trackersBoxes = new List<Box>();
            foreach (var tracker in trackers)
                trackersBoxes.Add(tracker.GetState());

            var costMatrix = getCostMatrix(detections, trackersBoxes);
            var columnMatches = Solver.Solve(costMatrix, true).ColumnAssignment;

            var matches = new List<List<int>>();
            var unmatchedDetections = new List<Box>();

            for (int i = 0;i<columnMatches.Length; i++)
                if (columnMatches[i] == -1 || costMatrix[i, columnMatches[i]] < iouThresh)
                    unmatchedDetections.Add(detections[i]);
                else
                    matches.Add(new List<int>() {i, columnMatches[i]});


            return Tuple.Create(matches, unmatchedDetections);
        }

        private double[,] getCostMatrix(List<Box> detections, List<Box> trackersBoxes)
        {
            var costMatrix = new double[detections.Count, trackersBoxes.Count];
            for (int i = 0; i < detections.Count; i++)
                for (int j = 0; j < trackersBoxes.Count; j++)
                    costMatrix[i, j] = Utils.IOU(detections[i], trackersBoxes[j]);
            return costMatrix;
        }

    }
}
