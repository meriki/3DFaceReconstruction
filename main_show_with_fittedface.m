%% Load Model
load('Model_Shape_Sim.mat');
data_path = '../';
face_path = '../../300W-3D-Face/'; %Your 300W-3D-Face path

%% Load Sample

listing = dir('../../300W-3D-Face/HELEN/');
length = size(listing);
for i=3:length(1)
sample_name = strcat('HELEN/', listing(i).name);
disp([data_path sample_name(1:end-4) '.jpg']);
disp(sample_name);
img = imread([data_path sample_name(1:end-4) '.jpg']);
load([data_path sample_name]);
load([face_path sample_name]);
ProjectVertex = Fitted_Face;
target = DrawSolidHead(ProjectVertex, tri);
save(strcat(face_path,'fitted_helen/',listing(i).name),'target');
end
