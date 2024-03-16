[filename, pathname] = uigetfile({'*.mp4';'*.avi';'*.mov'}, 'Select a video file');
filePath = fullfile(pathname, filename);
videoFile = vision.VideoFileReader(filePath, 'ImageColorSpace', 'RGB', 'VideoOutputDataType', 'uint8');

outputVideoName =  strcat(filename(1:end-4), '_tracked.avi');
outputVideo = VideoWriter(outputVideoName, 'Uncompressed AVI');
open(outputVideo);

frame = step(videoFile);
figureHandle = figure;
imshow(frame);
h = drawrectangle('Label', 'Drag to select object', 'Color', 'r');
bbox = floor(h.Position);
x = bbox(1);
y = bbox(2);
w = bbox(3);
h = bbox(4);
track_window = [x, y, w, h];

roi = frame(y:y+h, x:x+w, :);
gray_roi = rgb2gray(roi);
roi_hist = imhist(gray_roi, 256);
roi_hist = roi_hist / sum(roi_hist);

objectLost = false;
bboxVisible = true;
similarityThreshold = 0.5;

while ~isDone(videoFile)
    frame = step(videoFile);
    grayFrame = rgb2gray(frame);
    


    if ~objectLost
        meanShift = inf;
        posX = track_window(1);
        posY = track_window(2);
        previousWeights = [];
        weightsChanged = true;
        
        while weightsChanged
            currentWindow = grayFrame(posY:min(posY+h-1, size(grayFrame, 1)), posX:min(posX+w-1, size(grayFrame, 2)));
            currentHist = imhist(currentWindow, 256);
            currentHist = currentHist / sum(currentHist);
            
            rho = sum(sqrt(roi_hist .* currentHist));
            
            if rho < similarityThreshold
                objectLost = true;
                bboxVisible = false;
                break;
            else
                bboxVisible = true;
                newWeights = zeros(size(currentWindow));
                for i = 1:256
                    newWeights(currentWindow == i-1) = sqrt(roi_hist(i) / currentHist(i));
                end
                
                if isempty(previousWeights)
                    weightsChanged = true;
                else
                    weightChangeThreshold = 0.01;
                    weightChange = sum(abs(newWeights - previousWeights), 'all') / numel(newWeights);
                    
                    weightsChanged = weightChange > weightChangeThreshold;
                end
                
                previousWeights = newWeights;
                
                if weightsChanged
                    [yIndices, xIndices] = find(~isnan(newWeights));
                    if isempty(yIndices) || isempty(xIndices)
                        break;
                    end
                    xWeighted = sum(xIndices .* newWeights(~isnan(newWeights))) / sum(newWeights(~isnan(newWeights)));
                    yWeighted = sum(yIndices .* newWeights(~isnan(newWeights))) / sum(newWeights(~isnan(newWeights)));
                    
                    meanShift = [xWeighted - (w/2), yWeighted - (h/2)];
                    
                    posX = posX + round(meanShift(1));
                    posY = posY + round(meanShift(2));
                    
                    posX = max(1, min(size(frame, 2) - w, posX));
                    posY = max(1, min(size(frame, 1) - h, posY));
                end
            end
        end
    end    

    imshow(frame);
    hold on;
    if bboxVisible
        rectangle('Position', track_window, 'EdgeColor', 'yellow', 'LineWidth', 2);
    end
    hold off;
    drawnow;
    
    writeVideo(outputVideo, getframe(gcf));

    if bboxVisible
        track_window = [posX, posY, w, h];
    end
end

close(figureHandle);
release(videoFile);
close(outputVideo);