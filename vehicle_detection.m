    % Load video of the given location
the_video = VideoReader("videos\video3.mp4");

    % (object_detector) Detect foreground from any image. 
    % (NumGaussian) Machine learning model used to train model to detect vehicles.
    % (NumTrainingFrames) limit the number of frames
Object_Detector = vision.ForegroundDetector('NumGaussians', 3, 'NumTrainingFrames', 50);

for i = 1:150
    %read frames from video
       frame = readFrame(the_video);
    %pass frame to object detector
        the_object =  step(Object_Detector,frame);
end

figure,subplot(221),imshow(frame), title("Video frame");
subplot(222), imshow(the_object),title("Objects detected in Frame");

    % Morphological Operation to remove Noise
Structure = strel('square', 3);
Noise_Free_Object = imopen(the_object, Structure);
subplot(223), imshow(Noise_Free_Object),title('Object After Removing Noise');


% vision.Blob Analysis() will Ignore blobs objects which are less then a specific value (150 in this case). 
% This will return coordinates of boxes. Then bounding boxes will be drawn on detected objects.
     Bounding_Box = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 150);
     
     the_Box = step(Bounding_Box, Noise_Free_Object);


    % Drawing Rectangle on objects
Detected_Car = insertShape(frame, 'Rectangle', the_Box, 'Color', 'green');

    %Counting number of boxes to calculate total number of vehicles in a frame.
Number_of_Cars = size(the_Box, 1);
 Detected_Car = insertText(Detected_Car, [10 10], Number_of_Cars, 'BoxOpacity', 1,'FontSize', 14);
    
subplot(224),imshow(Detected_Car), title('Detected Cars in a frame');

videoPlayer = vision.VideoPlayer('Name', 'Detected Cars');
videoPlayer.Position(3:4) = [650,400];

while hasFrame(the_video)
frame = readFrame(the_video); 
the_object = step(Object_Detector, frame); 
Noise_Free_Object = imopen(the_object, Structure); 
the_Box = step(Bounding_Box, Noise_Free_Object); 
Detected_Car = insertShape(frame, 'Rectangle', the_Box, 'Color', 'green'); 
Number_of_Cars = size(the_Box, 1); 
Detected_Car = insertText(Detected_Car, [100 100], Number_of_Cars, 'BoxOpacity', 1, 'FontSize', 34); 
Number_of_Cars=Number_of_Cars+1;
step(videoPlayer, Detected_Car);
end

disp(Number_of_Cars);