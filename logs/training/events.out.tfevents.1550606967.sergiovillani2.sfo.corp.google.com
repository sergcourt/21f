       �K"	  ���Abrain.Event:2�J�M�      ��	�W���A"��
t
input/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
1layer_1/weights1/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_1/weights1*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_1/weights1/Initializer/random_uniform/minConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7��*
dtype0*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/maxConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7�?*
dtype0*
_output_shapes
: 
�
9layer_1/weights1/Initializer/random_uniform/RandomUniformRandomUniform1layer_1/weights1/Initializer/random_uniform/shape*
T0*#
_class
loc:@layer_1/weights1*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
/layer_1/weights1/Initializer/random_uniform/subSub/layer_1/weights1/Initializer/random_uniform/max/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
layer_1/weights1
VariableV2*#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
layer_1/weights1/readIdentitylayer_1/weights1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
!layer_1/biases1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
layer_1/biases1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
z
layer_1/biases1/readIdentitylayer_1/biases1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*
T0*'
_output_shapes
:���������
S
layer_1/ReluRelulayer_1/add*
T0*'
_output_shapes
:���������
�
1layer_2/weights2/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@layer_2/weights2*
valueB"      
�
/layer_2/weights2/Initializer/random_uniform/minConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/maxConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_2/weights2/Initializer/random_uniform/RandomUniformRandomUniform1layer_2/weights2/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_2/weights2*
seed2 
�
/layer_2/weights2/Initializer/random_uniform/subSub/layer_2/weights2/Initializer/random_uniform/max/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
layer_2/weights2
VariableV2*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
layer_2/weights2/readIdentitylayer_2/weights2*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
!layer_2/biases2/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
layer_2/biases2
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:
�
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
z
layer_2/biases2/readIdentitylayer_2/biases2*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
j
layer_2/addAddlayer_2/MatMullayer_2/biases2/read*
T0*'
_output_shapes
:���������
S
layer_2/ReluRelulayer_2/add*
T0*'
_output_shapes
:���������
�
1layer_3/weights3/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_3/weights3*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_3/weights3/Initializer/random_uniform/minConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/maxConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_3/weights3/Initializer/random_uniform/RandomUniformRandomUniform1layer_3/weights3/Initializer/random_uniform/shape*
T0*#
_class
loc:@layer_3/weights3*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/mulMul9layer_3/weights3/Initializer/random_uniform/RandomUniform/layer_3/weights3/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
+layer_3/weights3/Initializer/random_uniformAdd/layer_3/weights3/Initializer/random_uniform/mul/layer_3/weights3/Initializer/random_uniform/min*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
layer_3/weights3
VariableV2*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:*
dtype0*
_output_shapes

:
�
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
layer_3/weights3/readIdentitylayer_3/weights3*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
!layer_3/biases3/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_3/biases3*
valueB*    
�
layer_3/biases3
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
z
layer_3/biases3/readIdentitylayer_3/biases3*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
�
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_3/addAddlayer_3/MatMullayer_3/biases3/read*'
_output_shapes
:���������*
T0
S
layer_3/ReluRelulayer_3/add*
T0*'
_output_shapes
:���������
�
0output/weights4/Initializer/random_uniform/shapeConst*"
_class
loc:@output/weights4*
valueB"      *
dtype0*
_output_shapes
:
�
.output/weights4/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@output/weights4*
valueB
 *qĜ�
�
.output/weights4/Initializer/random_uniform/maxConst*"
_class
loc:@output/weights4*
valueB
 *qĜ?*
dtype0*
_output_shapes
: 
�
8output/weights4/Initializer/random_uniform/RandomUniformRandomUniform0output/weights4/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*"
_class
loc:@output/weights4*
seed2 
�
.output/weights4/Initializer/random_uniform/subSub.output/weights4/Initializer/random_uniform/max.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes
: 
�
.output/weights4/Initializer/random_uniform/mulMul8output/weights4/Initializer/random_uniform/RandomUniform.output/weights4/Initializer/random_uniform/sub*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
output/weights4
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4
�
output/weights4/AssignAssignoutput/weights4*output/weights4/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
~
output/weights4/readIdentityoutput/weights4*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
 output/biases4/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
output/biases4
VariableV2*!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
output/biases4/AssignAssignoutput/biases4 output/biases4/Initializer/zeros*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
w
output/biases4/readIdentityoutput/biases4*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
output/MatMulMatMullayer_3/Reluoutput/weights4/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
g

output/addAddoutput/MatMuloutput/biases4/read*
T0*'
_output_shapes
:���������
s
cost/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
{
cost/SquaredDifferenceSquaredDifference
output/addcost/Placeholder*
T0*'
_output_shapes
:���������
[

cost/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
}
,train/gradients/cost/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/cost/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
z
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
|
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/cost/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/cost/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
p
&train/gradients/cost/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
'train/gradients/cost/Mean_grad/floordivFloorDiv#train/gradients/cost/Mean_grad/Prod&train/gradients/cost/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
#train/gradients/cost/Mean_grad/CastCast'train/gradients/cost/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*
T0*'
_output_shapes
:���������
{
1train/gradients/cost/SquaredDifference_grad/ShapeShape
output/add*
T0*
out_type0*
_output_shapes
:
�
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
T0*
out_type0*
_output_shapes
:
�
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
/train/gradients/cost/SquaredDifference_grad/mulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/subSub
output/addcost/Placeholder'^train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
<train/gradients/cost/SquaredDifference_grad/tuple/group_depsNoOp0^train/gradients/cost/SquaredDifference_grad/Neg4^train/gradients/cost/SquaredDifference_grad/Reshape
�
Dtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/cost/SquaredDifference_grad/Reshape=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*F
_class<
:8loc:@train/gradients/cost/SquaredDifference_grad/Reshape
�
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg*'
_output_shapes
:���������
r
%train/gradients/output/add_grad/ShapeShapeoutput/MatMul*
_output_shapes
:*
T0*
out_type0
q
'train/gradients/output/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
5train/gradients/output/add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/add_grad/Shape'train/gradients/output/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
0train/gradients/output/add_grad/tuple/group_depsNoOp(^train/gradients/output/add_grad/Reshape*^train/gradients/output/add_grad/Reshape_1
�
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape*'
_output_shapes
:���������
�
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
_output_shapes
:*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
�
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul*'
_output_shapes
:���������
�
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_3/add_grad/ShapeShapelayer_3/MatMul*
_output_shapes
:*
T0*
out_type0
r
(train/gradients/layer_3/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_3/add_grad/ReshapeReshape$train/gradients/layer_3/add_grad/Sum&train/gradients/layer_3/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
&train/gradients/layer_3/add_grad/Sum_1Sum*train/gradients/layer_3/Relu_grad/ReluGrad8train/gradients/layer_3/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_3/add_grad/Reshape_1Reshape&train/gradients/layer_3/add_grad/Sum_1(train/gradients/layer_3/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_3/add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/add_grad/Reshape+^train/gradients/layer_3/add_grad/Reshape_1
�
9train/gradients/layer_3/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/add_grad/Reshape2^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_3/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1
�
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_3/MatMul_grad/tuple/control_dependencylayer_2/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
_output_shapes
:*
T0*
out_type0
r
(train/gradients/layer_2/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
1train/gradients/layer_2/add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/add_grad/Reshape+^train/gradients/layer_2/add_grad/Reshape_1
�
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
�
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_1/add_grad/ShapeShapelayer_1/MatMul*
_output_shapes
:*
T0*
out_type0
r
(train/gradients/layer_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
�
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
�
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul
�
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1*
_output_shapes

:
�
train/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *"
_class
loc:@layer_1/biases1*
valueB
 *fff?
�
train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
z
train/beta1_power/readIdentitytrain/beta1_power*
_output_shapes
: *
T0*"
_class
loc:@layer_1/biases1
�
train/beta2_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape: *
dtype0*
_output_shapes
: 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@layer_1/biases1
z
train/beta2_power/readIdentitytrain/beta2_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
-train/layer_1/weights1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_1/weights1*
valueB*    
�
train/layer_1/weights1/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container 
�
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_1/weights1*
valueB*    
�
train/layer_1/weights1/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:
�
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
"train/layer_1/weights1/Adam_1/readIdentitytrain/layer_1/weights1/Adam_1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
,train/layer_1/biases1/Adam/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:
�
!train/layer_1/biases1/Adam/AssignAssigntrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
train/layer_1/biases1/Adam/readIdentitytrain/layer_1/biases1/Adam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_1/biases1*
valueB*    
�
train/layer_1/biases1/Adam_1
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:
�
#train/layer_1/biases1/Adam_1/AssignAssigntrain/layer_1/biases1/Adam_1.train/layer_1/biases1/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
!train/layer_1/biases1/Adam_1/readIdentitytrain/layer_1/biases1/Adam_1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
-train/layer_2/weights2/Adam/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:
�
"train/layer_2/weights2/Adam/AssignAssigntrain/layer_2/weights2/Adam-train/layer_2/weights2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
 train/layer_2/weights2/Adam/readIdentitytrain/layer_2/weights2/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
/train/layer_2/weights2/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:
�
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
"train/layer_2/weights2/Adam_1/readIdentitytrain/layer_2/weights2/Adam_1*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
,train/layer_2/biases2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_2/biases2*
valueB*    
�
train/layer_2/biases2/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container 
�
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
train/layer_2/biases2/Adam/readIdentitytrain/layer_2/biases2/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_2/biases2
�
.train/layer_2/biases2/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam_1
VariableV2*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:
�
#train/layer_2/biases2/Adam_1/AssignAssigntrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@layer_2/biases2
�
-train/layer_3/weights3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_3/weights3*
valueB*    
�
train/layer_3/weights3/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container 
�
"train/layer_3/weights3/Adam/AssignAssigntrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
 train/layer_3/weights3/Adam/readIdentitytrain/layer_3/weights3/Adam*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_3/weights3*
valueB*    
�
train/layer_3/weights3/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
"train/layer_3/weights3/Adam_1/readIdentitytrain/layer_3/weights3/Adam_1*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
,train/layer_3/biases3/Adam/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
!train/layer_3/biases3/Adam/AssignAssigntrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
train/layer_3/biases3/Adam/readIdentitytrain/layer_3/biases3/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
�
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_3/biases3*
valueB*    
�
train/layer_3/biases3/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3
�
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
,train/output/weights4/Adam/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:
�
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
train/output/weights4/Adam/readIdentitytrain/output/weights4/Adam*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
.train/output/weights4/Adam_1/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:
�
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
!train/output/weights4/Adam_1/readIdentitytrain/output/weights4/Adam_1*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
+train/output/biases4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
�
train/output/biases4/Adam
VariableV2*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:
�
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/output/biases4/Adam/readIdentitytrain/output/biases4/Adam*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
-train/output/biases4/Adam_1/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
train/output/biases4/Adam_1
VariableV2*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:
�
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
 train/output/biases4/Adam_1/readIdentitytrain/output/biases4/Adam_1*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
]
train/Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_1/weights1*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@layer_1/biases1
�
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_2/weights2*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_2/biases2/ApplyAdam	ApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@layer_2/biases2
�
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_3/weights3*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_3/biases3*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@output/weights4*
use_nesterov( *
_output_shapes

:
�
*train/Adam/update_output/biases4/ApplyAdam	ApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
T0*!
_class
loc:@output/biases4*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*"
_class
loc:@layer_1/biases1
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam
n
logging/current_cost/tagsConst*
dtype0*
_output_shapes
: *%
valueB Blogging/current_cost
l
logging/current_costScalarSummarylogging/current_cost/tags	cost/Mean*
T0*
_output_shapes
: 
a
logging/Merge/MergeSummaryMergeSummarylogging/current_cost*
N*
_output_shapes
: 
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
�
save/SaveV2/tensor_namesConst*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslayer_1/biases1layer_1/weights1layer_2/biases2layer_2/weights2layer_3/biases3layer_3/weights3output/biases4output/weights4train/beta1_powertrain/beta2_powertrain/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/output/biases4/Adamtrain/output/biases4/Adam_1train/output/weights4/Adamtrain/output/weights4/Adam_1*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
�
save/AssignAssignlayer_1/biases1save/RestoreV2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_2Assignlayer_2/biases2save/RestoreV2:2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_7Assignoutput/weights4save/RestoreV2:7*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
save/Assign_8Assigntrain/beta1_powersave/RestoreV2:8*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_10Assigntrain/layer_1/biases1/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_11Assigntrain/layer_1/biases1/Adam_1save/RestoreV2:11*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_13Assigntrain/layer_1/weights1/Adam_1save/RestoreV2:13*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_20Assigntrain/layer_3/weights3/Adamsave/RestoreV2:20*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_22Assigntrain/output/biases4/Adamsave/RestoreV2:22*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
save/Assign_25Assigntrain/output/weights4/Adam_1save/RestoreV2:25*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign"�4ф�     ��d]	DYƝ�AJ܉
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.12.02v1.12.0-0-ga6d8ffae09��
t
input/PlaceholderPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
�
1layer_1/weights1/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_1/weights1*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_1/weights1/Initializer/random_uniform/minConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7��*
dtype0*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/maxConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7�?*
dtype0*
_output_shapes
: 
�
9layer_1/weights1/Initializer/random_uniform/RandomUniformRandomUniform1layer_1/weights1/Initializer/random_uniform/shape*
T0*#
_class
loc:@layer_1/weights1*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
/layer_1/weights1/Initializer/random_uniform/subSub/layer_1/weights1/Initializer/random_uniform/max/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
layer_1/weights1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container 
�
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
layer_1/weights1/readIdentitylayer_1/weights1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
!layer_1/biases1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
layer_1/biases1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1
�
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
z
layer_1/biases1/readIdentitylayer_1/biases1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
j
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*
T0*'
_output_shapes
:���������
S
layer_1/ReluRelulayer_1/add*
T0*'
_output_shapes
:���������
�
1layer_2/weights2/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_2/weights2*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_2/weights2/Initializer/random_uniform/minConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_2/weights2*
valueB
 *׳]?
�
9layer_2/weights2/Initializer/random_uniform/RandomUniformRandomUniform1layer_2/weights2/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_2/weights2*
seed2 
�
/layer_2/weights2/Initializer/random_uniform/subSub/layer_2/weights2/Initializer/random_uniform/max/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
layer_2/weights2
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container 
�
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
layer_2/weights2/readIdentitylayer_2/weights2*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
!layer_2/biases2/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
layer_2/biases2
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:
�
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
z
layer_2/biases2/readIdentitylayer_2/biases2*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
j
layer_2/addAddlayer_2/MatMullayer_2/biases2/read*'
_output_shapes
:���������*
T0
S
layer_2/ReluRelulayer_2/add*
T0*'
_output_shapes
:���������
�
1layer_3/weights3/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@layer_3/weights3*
valueB"      
�
/layer_3/weights3/Initializer/random_uniform/minConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_3/weights3*
valueB
 *׳]?
�
9layer_3/weights3/Initializer/random_uniform/RandomUniformRandomUniform1layer_3/weights3/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_3/weights3*
seed2 
�
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/mulMul9layer_3/weights3/Initializer/random_uniform/RandomUniform/layer_3/weights3/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
+layer_3/weights3/Initializer/random_uniformAdd/layer_3/weights3/Initializer/random_uniform/mul/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
layer_3/weights3
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
layer_3/weights3/readIdentitylayer_3/weights3*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
!layer_3/biases3/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_3/biases3*
valueB*    
�
layer_3/biases3
VariableV2*"
_class
loc:@layer_3/biases3*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
z
layer_3/biases3/readIdentitylayer_3/biases3*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j
layer_3/addAddlayer_3/MatMullayer_3/biases3/read*
T0*'
_output_shapes
:���������
S
layer_3/ReluRelulayer_3/add*
T0*'
_output_shapes
:���������
�
0output/weights4/Initializer/random_uniform/shapeConst*"
_class
loc:@output/weights4*
valueB"      *
dtype0*
_output_shapes
:
�
.output/weights4/Initializer/random_uniform/minConst*"
_class
loc:@output/weights4*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
�
.output/weights4/Initializer/random_uniform/maxConst*"
_class
loc:@output/weights4*
valueB
 *qĜ?*
dtype0*
_output_shapes
: 
�
8output/weights4/Initializer/random_uniform/RandomUniformRandomUniform0output/weights4/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*"
_class
loc:@output/weights4*
seed2 
�
.output/weights4/Initializer/random_uniform/subSub.output/weights4/Initializer/random_uniform/max.output/weights4/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@output/weights4
�
.output/weights4/Initializer/random_uniform/mulMul8output/weights4/Initializer/random_uniform/RandomUniform.output/weights4/Initializer/random_uniform/sub*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
output/weights4
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container 
�
output/weights4/AssignAssignoutput/weights4*output/weights4/Initializer/random_uniform*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
~
output/weights4/readIdentityoutput/weights4*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
 output/biases4/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
output/biases4
VariableV2*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:
�
output/biases4/AssignAssignoutput/biases4 output/biases4/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
w
output/biases4/readIdentityoutput/biases4*
_output_shapes
:*
T0*!
_class
loc:@output/biases4
�
output/MatMulMatMullayer_3/Reluoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
g

output/addAddoutput/MatMuloutput/biases4/read*
T0*'
_output_shapes
:���������
s
cost/PlaceholderPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
{
cost/SquaredDifferenceSquaredDifference
output/addcost/Placeholder*
T0*'
_output_shapes
:���������
[

cost/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
}
,train/gradients/cost/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/cost/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
z
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
|
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/cost/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
n
$train/gradients/cost/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
p
&train/gradients/cost/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
'train/gradients/cost/Mean_grad/floordivFloorDiv#train/gradients/cost/Mean_grad/Prod&train/gradients/cost/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
#train/gradients/cost/Mean_grad/CastCast'train/gradients/cost/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*
T0*'
_output_shapes
:���������
{
1train/gradients/cost/SquaredDifference_grad/ShapeShape
output/add*
_output_shapes
:*
T0*
out_type0
�
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
_output_shapes
:*
T0*
out_type0
�
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
�
/train/gradients/cost/SquaredDifference_grad/mulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/subSub
output/addcost/Placeholder'^train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
<train/gradients/cost/SquaredDifference_grad/tuple/group_depsNoOp0^train/gradients/cost/SquaredDifference_grad/Neg4^train/gradients/cost/SquaredDifference_grad/Reshape
�
Dtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/cost/SquaredDifference_grad/Reshape=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/cost/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg*'
_output_shapes
:���������
r
%train/gradients/output/add_grad/ShapeShapeoutput/MatMul*
T0*
out_type0*
_output_shapes
:
q
'train/gradients/output/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
5train/gradients/output/add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/add_grad/Shape'train/gradients/output/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
0train/gradients/output/add_grad/tuple/group_depsNoOp(^train/gradients/output/add_grad/Reshape*^train/gradients/output/add_grad/Reshape_1
�
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape*'
_output_shapes
:���������
�
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1*
_output_shapes
:
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
�
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul*'
_output_shapes
:���������
�
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1
�
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_3/add_grad/ShapeShapelayer_3/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_3/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_3/add_grad/ReshapeReshape$train/gradients/layer_3/add_grad/Sum&train/gradients/layer_3/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_3/add_grad/Sum_1Sum*train/gradients/layer_3/Relu_grad/ReluGrad8train/gradients/layer_3/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_3/add_grad/Reshape_1Reshape&train/gradients/layer_3/add_grad/Sum_1(train/gradients/layer_3/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
1train/gradients/layer_3/add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/add_grad/Reshape+^train/gradients/layer_3/add_grad/Reshape_1
�
9train/gradients/layer_3/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/add_grad/Reshape2^train/gradients/layer_3/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@train/gradients/layer_3/add_grad/Reshape
�
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_3/MatMul_grad/tuple/control_dependencylayer_2/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
_output_shapes
:*
T0*
out_type0
r
(train/gradients/layer_2/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_2/add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/add_grad/Reshape+^train/gradients/layer_2/add_grad/Reshape_1
�
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
�
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_1/add_grad/ShapeShapelayer_1/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
�
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
�
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1*
_output_shapes

:
�
train/beta1_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
train/beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@layer_1/biases1
z
train/beta1_power/readIdentitytrain/beta1_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2*"
_class
loc:@layer_1/biases1*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
z
train/beta2_power/readIdentitytrain/beta2_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
-train/layer_1/weights1/Adam/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container 
�
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_1/weights1*
valueB*    
�
train/layer_1/weights1/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:
�
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
"train/layer_1/weights1/Adam_1/readIdentitytrain/layer_1/weights1/Adam_1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
,train/layer_1/biases1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_1/biases1*
valueB*    
�
train/layer_1/biases1/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container 
�
!train/layer_1/biases1/Adam/AssignAssigntrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
train/layer_1/biases1/Adam/readIdentitytrain/layer_1/biases1/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
�
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam_1
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:
�
#train/layer_1/biases1/Adam_1/AssignAssigntrain/layer_1/biases1/Adam_1.train/layer_1/biases1/Adam_1/Initializer/zeros*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
�
!train/layer_1/biases1/Adam_1/readIdentitytrain/layer_1/biases1/Adam_1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
-train/layer_2/weights2/Adam/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:
�
"train/layer_2/weights2/Adam/AssignAssigntrain/layer_2/weights2/Adam-train/layer_2/weights2/Adam/Initializer/zeros*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
 train/layer_2/weights2/Adam/readIdentitytrain/layer_2/weights2/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
/train/layer_2/weights2/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:
�
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
"train/layer_2/weights2/Adam_1/readIdentitytrain/layer_2/weights2/Adam_1*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
,train/layer_2/biases2/Adam/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam
VariableV2*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:
�
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
train/layer_2/biases2/Adam/readIdentitytrain/layer_2/biases2/Adam*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
.train/layer_2/biases2/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:
�
#train/layer_2/biases2/Adam_1/AssignAssigntrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
-train/layer_3/weights3/Adam/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3
�
"train/layer_3/weights3/Adam/AssignAssigntrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
 train/layer_3/weights3/Adam/readIdentitytrain/layer_3/weights3/Adam*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
"train/layer_3/weights3/Adam_1/readIdentitytrain/layer_3/weights3/Adam_1*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
,train/layer_3/biases3/Adam/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam
VariableV2*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:*
dtype0*
_output_shapes
:
�
!train/layer_3/biases3/Adam/AssignAssigntrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
train/layer_3/biases3/Adam/readIdentitytrain/layer_3/biases3/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
�
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_3/biases3*
valueB*    
�
train/layer_3/biases3/Adam_1
VariableV2*"
_class
loc:@layer_3/biases3*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
,train/output/weights4/Adam/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam
VariableV2*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:
�
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
train/output/weights4/Adam/readIdentitytrain/output/weights4/Adam*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
.train/output/weights4/Adam_1/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:
�
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
!train/output/weights4/Adam_1/readIdentitytrain/output/weights4/Adam_1*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
+train/output/biases4/Adam/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
train/output/biases4/Adam
VariableV2*!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
train/output/biases4/Adam/readIdentitytrain/output/biases4/Adam*
_output_shapes
:*
T0*!
_class
loc:@output/biases4
�
-train/output/biases4/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
�
train/output/biases4/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:
�
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
 train/output/biases4/Adam_1/readIdentitytrain/output/biases4/Adam_1*
_output_shapes
:*
T0*!
_class
loc:@output/biases4
]
train/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_1/weights1*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_1/biases1*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@layer_2/weights2
�
+train/Adam/update_layer_2/biases2/ApplyAdam	ApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_2/biases2*
use_nesterov( *
_output_shapes
:
�
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_3/weights3*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_3/biases3*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
T0*"
_class
loc:@output/weights4*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
*train/Adam/update_output/biases4/ApplyAdam	ApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@output/biases4*
use_nesterov( *
_output_shapes
:
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam
n
logging/current_cost/tagsConst*%
valueB Blogging/current_cost*
dtype0*
_output_shapes
: 
l
logging/current_costScalarSummarylogging/current_cost/tags	cost/Mean*
T0*
_output_shapes
: 
a
logging/Merge/MergeSummaryMergeSummarylogging/current_cost*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1
�
save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslayer_1/biases1layer_1/weights1layer_2/biases2layer_2/weights2layer_3/biases3layer_3/weights3output/biases4output/weights4train/beta1_powertrain/beta2_powertrain/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/output/biases4/Adamtrain/output/biases4/Adam_1train/output/weights4/Adamtrain/output/weights4/Adam_1*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
�
save/AssignAssignlayer_1/biases1save/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_2Assignlayer_2/biases2save/RestoreV2:2*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_7Assignoutput/weights4save/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/Assign_8Assigntrain/beta1_powersave/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_10Assigntrain/layer_1/biases1/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_11Assigntrain/layer_1/biases1/Adam_1save/RestoreV2:11*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_13Assigntrain/layer_1/weights1/Adam_1save/RestoreV2:13*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_20Assigntrain/layer_3/weights3/Adamsave/RestoreV2:20*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_22Assigntrain/output/biases4/Adamsave/RestoreV2:22*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/Assign_25Assigntrain/output/weights4/Adam_1save/RestoreV2:25*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign""�
trainable_variables��
w
layer_1/weights1:0layer_1/weights1/Assignlayer_1/weights1/read:02-layer_1/weights1/Initializer/random_uniform:08
j
layer_1/biases1:0layer_1/biases1/Assignlayer_1/biases1/read:02#layer_1/biases1/Initializer/zeros:08
w
layer_2/weights2:0layer_2/weights2/Assignlayer_2/weights2/read:02-layer_2/weights2/Initializer/random_uniform:08
j
layer_2/biases2:0layer_2/biases2/Assignlayer_2/biases2/read:02#layer_2/biases2/Initializer/zeros:08
w
layer_3/weights3:0layer_3/weights3/Assignlayer_3/weights3/read:02-layer_3/weights3/Initializer/random_uniform:08
j
layer_3/biases3:0layer_3/biases3/Assignlayer_3/biases3/read:02#layer_3/biases3/Initializer/zeros:08
s
output/weights4:0output/weights4/Assignoutput/weights4/read:02,output/weights4/Initializer/random_uniform:08
f
output/biases4:0output/biases4/Assignoutput/biases4/read:02"output/biases4/Initializer/zeros:08"'
	summaries

logging/current_cost:0"
train_op


train/Adam"�
	variables��
w
layer_1/weights1:0layer_1/weights1/Assignlayer_1/weights1/read:02-layer_1/weights1/Initializer/random_uniform:08
j
layer_1/biases1:0layer_1/biases1/Assignlayer_1/biases1/read:02#layer_1/biases1/Initializer/zeros:08
w
layer_2/weights2:0layer_2/weights2/Assignlayer_2/weights2/read:02-layer_2/weights2/Initializer/random_uniform:08
j
layer_2/biases2:0layer_2/biases2/Assignlayer_2/biases2/read:02#layer_2/biases2/Initializer/zeros:08
w
layer_3/weights3:0layer_3/weights3/Assignlayer_3/weights3/read:02-layer_3/weights3/Initializer/random_uniform:08
j
layer_3/biases3:0layer_3/biases3/Assignlayer_3/biases3/read:02#layer_3/biases3/Initializer/zeros:08
s
output/weights4:0output/weights4/Assignoutput/weights4/read:02,output/weights4/Initializer/random_uniform:08
f
output/biases4:0output/biases4/Assignoutput/biases4/read:02"output/biases4/Initializer/zeros:08
l
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:02!train/beta1_power/initial_value:0
l
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:02!train/beta2_power/initial_value:0
�
train/layer_1/weights1/Adam:0"train/layer_1/weights1/Adam/Assign"train/layer_1/weights1/Adam/read:02/train/layer_1/weights1/Adam/Initializer/zeros:0
�
train/layer_1/weights1/Adam_1:0$train/layer_1/weights1/Adam_1/Assign$train/layer_1/weights1/Adam_1/read:021train/layer_1/weights1/Adam_1/Initializer/zeros:0
�
train/layer_1/biases1/Adam:0!train/layer_1/biases1/Adam/Assign!train/layer_1/biases1/Adam/read:02.train/layer_1/biases1/Adam/Initializer/zeros:0
�
train/layer_1/biases1/Adam_1:0#train/layer_1/biases1/Adam_1/Assign#train/layer_1/biases1/Adam_1/read:020train/layer_1/biases1/Adam_1/Initializer/zeros:0
�
train/layer_2/weights2/Adam:0"train/layer_2/weights2/Adam/Assign"train/layer_2/weights2/Adam/read:02/train/layer_2/weights2/Adam/Initializer/zeros:0
�
train/layer_2/weights2/Adam_1:0$train/layer_2/weights2/Adam_1/Assign$train/layer_2/weights2/Adam_1/read:021train/layer_2/weights2/Adam_1/Initializer/zeros:0
�
train/layer_2/biases2/Adam:0!train/layer_2/biases2/Adam/Assign!train/layer_2/biases2/Adam/read:02.train/layer_2/biases2/Adam/Initializer/zeros:0
�
train/layer_2/biases2/Adam_1:0#train/layer_2/biases2/Adam_1/Assign#train/layer_2/biases2/Adam_1/read:020train/layer_2/biases2/Adam_1/Initializer/zeros:0
�
train/layer_3/weights3/Adam:0"train/layer_3/weights3/Adam/Assign"train/layer_3/weights3/Adam/read:02/train/layer_3/weights3/Adam/Initializer/zeros:0
�
train/layer_3/weights3/Adam_1:0$train/layer_3/weights3/Adam_1/Assign$train/layer_3/weights3/Adam_1/read:021train/layer_3/weights3/Adam_1/Initializer/zeros:0
�
train/layer_3/biases3/Adam:0!train/layer_3/biases3/Adam/Assign!train/layer_3/biases3/Adam/read:02.train/layer_3/biases3/Adam/Initializer/zeros:0
�
train/layer_3/biases3/Adam_1:0#train/layer_3/biases3/Adam_1/Assign#train/layer_3/biases3/Adam_1/read:020train/layer_3/biases3/Adam_1/Initializer/zeros:0
�
train/output/weights4/Adam:0!train/output/weights4/Adam/Assign!train/output/weights4/Adam/read:02.train/output/weights4/Adam/Initializer/zeros:0
�
train/output/weights4/Adam_1:0#train/output/weights4/Adam_1/Assign#train/output/weights4/Adam_1/read:020train/output/weights4/Adam_1/Initializer/zeros:0
�
train/output/biases4/Adam:0 train/output/biases4/Adam/Assign train/output/biases4/Adam/read:02-train/output/biases4/Adam/Initializer/zeros:0
�
train/output/biases4/Adam_1:0"train/output/biases4/Adam_1/Assign"train/output/biases4/Adam_1/read:02/train/output/biases4/Adam_1/Initializer/zeros:0$��A(       �pJ	�s˝�A*

logging/current_cost1�	=�`*       ����	M�˝�A*

logging/current_cost�a=�m�*       ����	��˝�A
*

logging/current_cost���<��3�*       ����	"̝�A*

logging/current_cost���<�OC�*       ����	^S̝�A*

logging/current_cost���<P���*       ����	��̝�A*

logging/current_cost�c�<�B*       ����	F�̝�A*

logging/current_cost�9�<^iR*       ����	��̝�A#*

logging/current_cost���<"��*       ����	�"͝�A(*

logging/current_cost�5�<���*       ����	X͝�A-*

logging/current_cost���<�
6O*       ����	8�͝�A2*

logging/current_cost���<�G*       ����	A�͝�A7*

logging/current_cost)�<���4*       ����	��͝�A<*

logging/current_costA��<����*       ����	r Ν�AA*

logging/current_costjL�<���*       ����	{RΝ�AF*

logging/current_cost��<���*       ����	k�Ν�AK*

logging/current_cost��<�>�*       ����	��Ν�AP*

logging/current_cost���<;s/]*       ����	��Ν�AU*

logging/current_cost`��<).<*       ����	�ϝ�AZ*

logging/current_cost]��<W��*       ����	�<ϝ�A_*

logging/current_cost���<B�Z*       ����	:nϝ�Ad*

logging/current_costv��<�"��*       ����	s�ϝ�Ai*

logging/current_cost��<��*       ����	��ϝ�An*

logging/current_costo��<J�+�*       ����	�Н�As*

logging/current_cost���<~��*       ����	�/Н�Ax*

logging/current_cost��<g�m*       ����	^Н�A}*

logging/current_cost���<$��+       ��K	�Н�A�*

logging/current_cost���<�W�+       ��K	g�Н�A�*

logging/current_cost���<��5+       ��K	eѝ�A�*

logging/current_cost���<���+       ��K	�Aѝ�A�*

logging/current_cost���<jط�+       ��K	�uѝ�A�*

logging/current_cost���<V�` +       ��K	��ѝ�A�*

logging/current_cost��<�^�+       ��K	��ѝ�A�*

logging/current_cost���<[��+       ��K	4ҝ�A�*

logging/current_cost���<��u+       ��K	�Jҝ�A�*

logging/current_cost���<��a+       ��K	�zҝ�A�*

logging/current_cost��<l��+       ��K	�ҝ�A�*

logging/current_costCj�<�]�g+       ��K	��ҝ�A�*

logging/current_costb��<`y��+       ��K	ӝ�A�*

logging/current_cost{*�<���+       ��K	�Dӝ�A�*

logging/current_cost�C�<GM�+       ��K	Ptӝ�A�*

logging/current_costR�<t�f�+       ��K	��ӝ�A�*

logging/current_cost\Y�<4W+       ��K	S�ӝ�A�*

logging/current_cost�<v�B+       ��K	 ԝ�A�*

logging/current_costU��<��R�+       ��K	�;ԝ�A�*

logging/current_cost�E�<�y؅+       ��K	�iԝ�A�*

logging/current_cost�.�<�m\�+       ��K	Řԝ�A�*

logging/current_costp¨<�)�+       ��K	��ԝ�A�*

logging/current_cost�#�<��w�+       ��K	1�ԝ�A�*

logging/current_cost0�<�ˠ2+       ��K	,՝�A�*

logging/current_cost)-�<���7+       ��K	v]՝�A�*

logging/current_cost�4�<nk�i+       ��K	�՝�A�*

logging/current_costBY�<�a�T+       ��K	I�՝�A�*

logging/current_costGj�<'"��+       ��K	��՝�A�*

logging/current_cost��<#͠0+       ��K	�+֝�A�*

logging/current_costLnt<�C�S+       ��K	�[֝�A�*

logging/current_cost��h<Bq�n+       ��K	 �֝�A�*

logging/current_cost�)^<����+       ��K	��֝�A�*

logging/current_cost��S<�+       ��K	��֝�A�*

logging/current_costJ<![-+       ��K	vם�A�*

logging/current_cost̠@<�!��+       ��K	MSם�A�*

logging/current_cost��7<lƑU+       ��K	y�ם�A�*

logging/current_cost�/<=�-Q+       ��K	U�ם�A�*

logging/current_costn
'<b�P8+       ��K	�ם�A�*

logging/current_costhk<��\+       ��K	�؝�A�*

logging/current_costBb<0l��+       ��K	F؝�A�*

logging/current_cost.�<��+       ��K	1x؝�A�*

logging/current_costf�<C��A+       ��K	��؝�A�*

logging/current_cost1A<��`�+       ��K	.�؝�A�*

logging/current_cost^� <MZϦ+       ��K	!ٝ�A�*

logging/current_cost���;"��+       ��K	(@ٝ�A�*

logging/current_cost(��;�,��+       ��K	�nٝ�A�*

logging/current_cost�I�;}���+       ��K	�ٝ�A�*

logging/current_cost\4�;_(�+       ��K	��ٝ�A�*

logging/current_costB��;IB��+       ��K	� ڝ�A�*

logging/current_cost���;[�'�+       ��K	7ڝ�A�*

logging/current_cost��;SZ�j+       ��K	�iڝ�A�*

logging/current_cost��;�E��+       ��K	;�ڝ�A�*

logging/current_costF�;����+       ��K	I�ڝ�A�*

logging/current_costS�;��+       ��K	W�ڝ�A�*

logging/current_cost�̳;�dD9+       ��K	3۝�A�*

logging/current_costC��;�pޫ+       ��K	�g۝�A�*

logging/current_cost���;��}+       ��K	�۝�A�*

logging/current_cost���;z�S+       ��K	D�۝�A�*

logging/current_cost{��;~��^+       ��K	�ܝ�A�*

logging/current_cost?��;���F+       ��K	�5ܝ�A�*

logging/current_cost�F�;fL+       ��K	�rܝ�A�*

logging/current_cost7Z�;{ewG+       ��K	A�ܝ�A�*

logging/current_cost���;��ja+       ��K	Z�ܝ�A�*

logging/current_coste�;<��+       ��K	�ݝ�A�*

logging/current_cost�`�;�v1+       ��K	jAݝ�A�*

logging/current_cost��;R	f�+       ��K	�rݝ�A�*

logging/current_cost�F�;�Uނ+       ��K	��ݝ�A�*

logging/current_cost�$�;��ژ+       ��K	.�ݝ�A�*

logging/current_cost$�;�ËQ+       ��K	"ޝ�A�*

logging/current_cost~8�;:�}�+       ��K	T;ޝ�A�*

logging/current_costYo�;vߕ�+       ��K	>kޝ�A�*

logging/current_costW��;3�.+       ��K	'�ޝ�A�*

logging/current_cost9 �;�IK�+       ��K	�ޝ�A�*

logging/current_cost���;nν�+       ��K	�ޝ�A�*

logging/current_cost��;ש1	+       ��K	W'ߝ�A�*

logging/current_costۗ�;�O�+       ��K	oUߝ�A�*

logging/current_cost�-�;E��+       ��K	��ߝ�A�*

logging/current_cost�ې;V@p(+       ��K	��ߝ�A�*

logging/current_cost4��;LI9R+       ��K	Y�ߝ�A�*

logging/current_costZ�;��*:+       ��K	����A�*

logging/current_cost1&�;���O+       ��K	�;���A�*

logging/current_costD��;� �+       ��K	�g���A�*

logging/current_costMˏ;���+       ��K	~����A�*

logging/current_cost��;�u+       ��K	����A�*

logging/current_cost�|�;6)�A+       ��K	f����A�*

logging/current_costY�;wP։+       ��K	q ��A�*

logging/current_costf8�;?�Z+       ��K	YN��A�*

logging/current_cost��;0�+       ��K	�|��A�*

logging/current_cost���;M���+       ��K	����A�*

logging/current_cost��;�}��+       ��K	����A�*

logging/current_cost-͎;��Q+       ��K	��A�*

logging/current_costԷ�;~!�$+       ��K	4��A�*

logging/current_cost0��;�G�X+       ��K	�c��A�*

logging/current_cost��;�6��+       ��K	����A�*

logging/current_costu��;wN��+       ��K	R���A�*

logging/current_cost~r�;ڲ��+       ��K	���A�*

logging/current_cost�d�;�U�+       ��K	9��A�*

logging/current_cost]X�;��a�+       ��K	1K��A�*

logging/current_cost�L�;�τ:+       ��K	�x��A�*

logging/current_cost�B�;?Uн+       ��K	̧��A�*

logging/current_cost99�;*�b+       ��K	����A�*

logging/current_cost�0�;���+       ��K	���A�*

logging/current_cost��;���:+       ��K	�1��A�*

logging/current_cost���;��qd+       ��K	5`��A�*

logging/current_cost.m�;��|+       ��K	����A�*

logging/current_costm1�;S��+       ��K	����A�*

logging/current_cost��;�_�+       ��K	P��A�*

logging/current_costI�;u�e�+       ��K	 N��A�*

logging/current_cost9ˌ;�R-+       ��K	����A�*

logging/current_cost᪌;H��+       ��K	{���A�*

logging/current_cost6��;W]�+       ��K	���A�*

logging/current_cost��;\D+       ��K	#5��A�*

logging/current_cost�{�;$�K�+       ��K	�f��A�*

logging/current_cost�q�;\`�\+       ��K	���A�*

logging/current_cost6h�;F��Y+       ��K	z���A�*

logging/current_cost�^�;b�n�+       ��K	�
��A�*

logging/current_cost�V�;y�]�+       ��K	�>��A�*

logging/current_costjO�;.O��+       ��K	yo��A�*

logging/current_cost�H�;��+       ��K	<���A�*

logging/current_cost�B�;��j+       ��K	5���A�*

logging/current_cost=�;��S�+       ��K	��A�*

logging/current_cost�7�;�F��+       ��K	�G��A�*

logging/current_cost	3�;;i;g+       ��K	�z��A�*

logging/current_cost�.�;�͋�+       ��K	ݱ��A�*

logging/current_cost�*�;5
�"+       ��K	���A�*

logging/current_cost�&�;'�P�+       ��K	L��A�*

logging/current_cost�#�;eȗ%+       ��K	�M��A�*

logging/current_costx �;~�	+       ��K	e���A�*

logging/current_cost��;�]+       ��K	���A�*

logging/current_cost�;mr�:+       ��K	����A�*

logging/current_cost��;I8��+       ��K	� ��A�*

logging/current_cost~�;�:�+       ��K	W��A�*

logging/current_costtۋ;�Z�+       ��K	��A�*

logging/current_cost���;�(8�+       ��K	����A�*

logging/current_cost���;���+       ��K	����A�*

logging/current_cost��;�m�+       ��K	'��A�*

logging/current_cost0i�;de+       ��K	�Z��A�*

logging/current_cost�Q�;�,3+       ��K	+���A�*

logging/current_cost�:�;�߹+       ��K	���A�*

logging/current_cost�&�;����+       ��K	����A�*

logging/current_cost��;2"-+       ��K	��A�*

logging/current_costM�;s�72+       ��K	Y��A�*

logging/current_cost��;fs\+       ��K	����A�*

logging/current_cost��;�'Yk+       ��K	n���A�*

logging/current_cost(֊;��+       ��K	i���A�*

logging/current_cost�Ɋ;�!E+       ��K	1��A�*

logging/current_cost"��;�o��+       ��K	�P��A�*

logging/current_costݳ�;�J\+       ��K	 ���A�*

logging/current_costv��;=�4t+       ��K	����A�*

logging/current_cost���;���(+       ��K	v���A�*

logging/current_cost���;qV��+       ��K	n��A�*

logging/current_cost���;j�^�+       ��K	_?��A�*

logging/current_cost��;^!�+       ��K	�p��A�*

logging/current_cost���;��K)+       ��K	����A�*

logging/current_cost���;&���+       ��K	~���A�*

logging/current_cost�{�;j�+       ��K	9��A�*

logging/current_costw�;<���+       ��K	�>��A�*

logging/current_costs�;��xf+       ��K	gq��A�*

logging/current_costYo�;C܎�+       ��K	���A�*

logging/current_costl�;�s*\+       ��K	.���A�*

logging/current_cost
i�;��+       ��K	���A�*

logging/current_costSf�;a}��+       ��K	B��A�*

logging/current_cost�c�;� }+       ��K	�~��A�*

logging/current_cost�a�;[���+       ��K	����A�*

logging/current_cost�_�;����+       ��K	f���A�*

logging/current_cost�]�;T=�S+       ��K	`%��A�*

logging/current_cost?\�;�35�+       ��K	�U��A�*

logging/current_cost�Z�;���+       ��K	U���A�*

logging/current_cost~Y�;�0G�+       ��K	C���A�*

logging/current_costTW�;>���+       ��K	����A�*

logging/current_cost�R�;���+       ��K	�+��A�*

logging/current_costN�;��Z+       ��K	�Z��A�*

logging/current_cost�J�;0	�M+       ��K	����A�*

logging/current_cost_G�;hR!M+       ��K	_���A�*

logging/current_costFD�;f�>�+       ��K	���A�*

logging/current_cost%A�;�p�9+       ��K	J9��A�*

logging/current_costS>�;�)�+       ��K	�z��A�*

logging/current_cost�;�;��s+       ��K	����A�*

logging/current_cost�9�;�!��+       ��K	|���A�*

logging/current_costh7�;+�	+       ��K	p@���A�*

logging/current_cost|5�;�r�f+       ��K	�o���A�*

logging/current_cost�3�;�(U
+       ��K	}����A�*

logging/current_cost/2�;��+       ��K	k����A�*

logging/current_cost�0�;�i�+       ��K	�����A�*

logging/current_costx/�;i��6+       ��K	�,���A�*

logging/current_costM.�;6��+       ��K	�]���A�*

logging/current_cost<-�;��+       ��K	]����A�*

logging/current_costG,�;@�	+       ��K	8����A�*

logging/current_costk+�;��6+       ��K	v����A�*

logging/current_cost�*�;�/2/+       ��K	����A�*

logging/current_cost�)�;��R�+       ��K	HQ���A�*

logging/current_costO)�;xfӛ+       ��K	�����A�*

logging/current_cost�(�;l��+       ��K	����A�*

logging/current_cost?(�;M���+       ��K	����A�*

logging/current_cost�'�;8Ď�+       ��K	+'���A�*

logging/current_cost\'�;T��+       ��K	t���A�*

logging/current_cost�&�;��n+       ��K	!����A�*

logging/current_cost�&�;�u
z+       ��K	�����A�*

logging/current_costZ&�;�Ųi+       ��K	���A�*

logging/current_cost&�;�N�y+       ��K	 R���A�*

logging/current_cost�%�;��Q�+       ��K	A����A�*

logging/current_cost�%�;SD+       ��K	����A�*

logging/current_costp%�;M�z,+       ��K	*����A�*

logging/current_costC%�;����+       ��K	���A�*

logging/current_cost%�;��+       ��K	a?���A�*

logging/current_cost�$�;�#+�+       ��K	�n���A�	*

logging/current_cost�$�;,��+       ��K	����A�	*

logging/current_cost�$�;ʧ�4+       ��K	w����A�	*

logging/current_cost�$�;�Jf�+       ��K	�����A�	*

logging/current_cost�$�;�,�+       ��K	�#���A�	*

logging/current_cost}$�;�_[�+       ��K	�S���A�	*

logging/current_costl$�;�n�7+       ��K	�����A�	*

logging/current_costY$�;kT�+       ��K	ɮ���A�	*

logging/current_costM$�;��`L+       ��K	C����A�	*

logging/current_cost@$�;�r+       ��K	�	���A�	*

logging/current_cost5$�;̛�+       ��K	�7���A�	*

logging/current_cost<$�;57+       ��K	Fc���A�	*

logging/current_costI$�;�O��+       ��K	V����A�	*

logging/current_cost=$�;F�]+       ��K	�����A�	*

logging/current_cost#$�;ZđO+       ��K	����A�	*

logging/current_costR$�;ǃ�+       ��K	�_���A�	*

logging/current_costV$�;+�0+       ��K	�����A�	*

logging/current_cost:$�;�doX+       ��K	�����A�	*

logging/current_cost$�;�{(+       ��K	�$���A�	*

logging/current_costM$�;�9g�+       ��K	�^���A�	*

logging/current_costG$�;�'��+       ��K	ɔ���A�	*

logging/current_cost$�;�&�)+       ��K	�����A�	*

logging/current_costU$�;��<(+       ��K	j����A�	*

logging/current_cost^$�;R��+       ��K	L5���A�	*

logging/current_cost�#�;Swo+       ��K	i���A�	*

logging/current_cost�#�;���+       ��K	����A�
*

logging/current_costK#�;�\+       ��K	�����A�
*

logging/current_cost#�;���+       ��K	k���A�
*

logging/current_cost�"�;KzT�+       ��K	�=���A�
*

logging/current_cost�"�;g�K+       ��K	�u���A�
*

logging/current_cost�"�;h>�+       ��K	æ���A�
*

logging/current_cost�"�;�0)+       ��K	�����A�
*

logging/current_cost�"�;4/�+       ��K	� ��A�
*

logging/current_cost*"�;0��'+       ��K	=_ ��A�
*

logging/current_cost�!�;��i�+       ��K	Z� ��A�
*

logging/current_costu!�;Q$��+       ��K	ҿ ��A�
*

logging/current_cost%!�;
<)�+       ��K	� ��A�
*

logging/current_cost!�;��^�+       ��K	�!��A�
*

logging/current_cost� �;kTj+       ��K	�Q��A�
*

logging/current_costt �;m)��+       ��K	>���A�
*

logging/current_costA �;}�V�+       ��K	���A�
*

logging/current_cost  �;L6+       ��K	����A�
*

logging/current_cost��;#��+       ��K	!��A�
*

logging/current_cost��;͙�.+       ��K	�H��A�
*

logging/current_costT�;�ʰ�+       ��K	�v��A�
*

logging/current_cost�;[��+       ��K	����A�
*

logging/current_costr�;�$��+       ��K	��A�
*

logging/current_cost��;W&C�+       ��K	�A��A�
*

logging/current_costn�;z	+       ��K	}���A�
*

logging/current_costr�;�
��+       ��K	���A�
*

logging/current_costR�;順�+       ��K	o��A�
*

logging/current_cost��;����+       ��K	U<��A�*

logging/current_cost��;�W�[+       ��K	tw��A�*

logging/current_costT�;�t+       ��K	y���A�*

logging/current_costB�;8.�#+       ��K	����A�*

logging/current_cost��;���/+       ��K	�*��A�*

logging/current_cost��;�e�-+       ��K	_��A�*

logging/current_cost��;M2/+       ��K	[���A�*

logging/current_cost��;��-+       ��K	����A�*

logging/current_cost��;��+       ��K	����A�*

logging/current_costq�;���+       ��K	$.��A�*

logging/current_cost��;��Z�+       ��K	�`��A�*

logging/current_cost��;�r܄+       ��K	����A�*

logging/current_costd�;�7+       ��K	+���A�*

logging/current_cost��;{&hJ+       ��K	t���A�*

logging/current_cost��;`a1H+       ��K	���A�*

logging/current_cost��;��!+       ��K	oP��A�*

logging/current_cost��;�h4+       ��K	�~��A�*

logging/current_cost��;'�g+       ��K	���A�*

logging/current_cost�
�;9�L-+       ��K	v���A�*

logging/current_cost�	�;�5+       ��K	~��A�*

logging/current_cost��;�Yݸ+       ��K	I4��A�*

logging/current_cost��;�?�+       ��K	6`��A�*

logging/current_cost��;X'i�+       ��K	����A�*

logging/current_costq�;т�d+       ��K	����A�*

logging/current_cost<�;[� 
+       ��K	3���A�*

logging/current_cost+�;��+       ��K	F&	��A�*

logging/current_costi�;��T+       ��K	�w	��A�*

logging/current_cost��;�?{�+       ��K	/�	��A�*

logging/current_costO �;��`�+       ��K	�	��A�*

logging/current_cost0��;H��+       ��K	'
��A�*

logging/current_costU��;5! �+       ��K	qI
��A�*

logging/current_cost���;�ڎ�+       ��K	�x
��A�*

logging/current_cost$��;L�++       ��K	��
��A�*

logging/current_cost|��;lɳ+       ��K	�
��A�*

logging/current_costR��;��i+       ��K	���A�*

logging/current_cost���;�Q��+       ��K	K3��A�*

logging/current_cost���;k�B�+       ��K	�_��A�*

logging/current_cost���;>vr+       ��K	b���A�*

logging/current_cost:��;-f��+       ��K	���A�*

logging/current_cost���;���]+       ��K	����A�*

logging/current_cost��;�(\+       ��K	���A�*

logging/current_cost��;�_m+       ��K	vI��A�*

logging/current_cost-�;K�b+       ��K	�u��A�*

logging/current_cost�;�"E�+       ��K	����A�*

logging/current_cost��;$���+       ��K	����A�*

logging/current_cost��;�W�=+       ��K	���A�*

logging/current_costz�;!��+       ��K	X?��A�*

logging/current_cost��;j��+       ��K	�s��A�*

logging/current_cost)�;!+       ��K	����A�*

logging/current_cost6�;#@S+       ��K	���A�*

logging/current_cost9�;��\+       ��K	
��A�*

logging/current_cost��;�I+       ��K	z0��A�*

logging/current_cost��;�8�+       ��K	�b��A�*

logging/current_cost��;��3�+       ��K	 ���A�*

logging/current_cost��;`+�+       ��K	����A�*

logging/current_cost^�;�ʭ>+       ��K	���A�*

logging/current_cost]�;���+       ��K	n"��A�*

logging/current_cost�; h�=+       ��K	uO��A�*

logging/current_cost��;[�k�+       ��K	U���A�*

logging/current_cost�މ;Ҡ��+       ��K	u���A�*

logging/current_cost�݉;8��j+       ��K	(���A�*

logging/current_cost�܉;�I�+       ��K	���A�*

logging/current_cost�ۉ;�T�+       ��K	wK��A�*

logging/current_cost^ډ;��//+       ��K	����A�*

logging/current_cost�ى;��T�+       ��K	����A�*

logging/current_cost0؉;�y��+       ��K	����A�*

logging/current_cost�׉;x�V�+       ��K	i��A�*

logging/current_cost ։;�^�/+       ��K	�<��A�*

logging/current_cost�Չ;;:~++       ��K	&j��A�*

logging/current_costԉ;�k�+       ��K	(���A�*

logging/current_costӉ; R��+       ��K	����A�*

logging/current_cost-҉;]9*G+       ��K	 ��A�*

logging/current_cost�Љ;8DF�+       ��K	0��A�*

logging/current_cost�ω;��_+       ��K	B^��A�*

logging/current_cost�Ή;��7+       ��K	e���A�*

logging/current_cost�͉;#2��+       ��K	����A�*

logging/current_cost�̉;��h�+       ��K	����A�*

logging/current_cost�ˉ;3���+       ��K	���A�*

logging/current_cost�ʉ;�W�X+       ��K	�J��A�*

logging/current_cost�ɉ;oLQf+       ��K	Ow��A�*

logging/current_cost�ȉ;��8�+       ��K	ɧ��A�*

logging/current_cost�ǉ;L��$+       ��K	����A�*

logging/current_costnƉ;���d+       ��K	���A�*

logging/current_cost�ŉ;�:�+       ��K	U<��A�*

logging/current_cost|ĉ;��S`+       ��K	`j��A�*

logging/current_costbÉ;eG��+       ��K	N���A�*

logging/current_costt;�FW,+       ��K	A���A�*

logging/current_costX��;�d��+       ��K	���A�*

logging/current_cost\��;�:"�+       ��K	� ��A�*

logging/current_cost���;Sca�+       ��K	�O��A�*

logging/current_coste��;-q]+       ��K	M{��A�*

logging/current_costi��;հJ�+       ��K	����A�*

logging/current_costH��;�A�+       ��K	����A�*

logging/current_costj��;��vs+       ��K	��A�*

logging/current_costj��;/�vU+       ��K	>4��A�*

logging/current_cost_��;���+       ��K	Xc��A�*

logging/current_costT��;8h($+       ��K	\���A�*

logging/current_cost_��;��5B+       ��K	����A�*

logging/current_cost`��; �h�+       ��K	Y���A�*

logging/current_costc��;��ϓ+       ��K	
��A�*

logging/current_costk��;P=��+       ��K	}C��A�*

logging/current_costr��;KJ��+       ��K	�r��A�*

logging/current_costy��;G-q�+       ��K	Q���A�*

logging/current_cost���;�!�@+       ��K	*���A�*

logging/current_cost���;�+       ��K	����A�*

logging/current_cost���;<�M +       ��K	J)��A�*

logging/current_cost���;�Fg�+       ��K	HW��A�*

logging/current_cost���;�mm�+       ��K	˅��A�*

logging/current_cost���;t��+       ��K	���A�*

logging/current_cost���;UI�:+       ��K	e���A�*

logging/current_cost���;6\�M+       ��K	���A�*

logging/current_cost���;�O��+       ��K	f@��A�*

logging/current_cost���;���-+       ��K	�m��A�*

logging/current_cost���;��3�+       ��K	,���A�*

logging/current_cost���;\��+       ��K	����A�*

logging/current_cost���;�Y7+       ��K	3��A�*

logging/current_cost���;η+       ��K	l2��A�*

logging/current_cost���;��+       ��K	&a��A�*

logging/current_cost���;X��+       ��K	����A�*

logging/current_cost���;���+       ��K	���A�*

logging/current_cost���;o7+       ��K	����A�*

logging/current_cost���;1G��+       ��K	i��A�*

logging/current_cost���;��I�+       ��K	cJ��A�*

logging/current_cost���;��¥+       ��K	)x��A�*

logging/current_costz��;�*�+       ��K	���A�*

logging/current_costn��;*� �+       ��K	����A�*

logging/current_costc��;�R�+       ��K	���A�*

logging/current_costR��;����+       ��K	�2��A�*

logging/current_costC��;Vk��+       ��K	�_��A�*

logging/current_cost1��;�x�+       ��K	����A�*

logging/current_cost��;#0r�+       ��K	\���A�*

logging/current_cost
��;��+       ��K	E���A�*

logging/current_cost���;Hx��+       ��K	h��A�*

logging/current_costߒ�;�x��+       ��K	'K��A�*

logging/current_costő�;
�4+       ��K	C|��A�*

logging/current_cost���;0gK8+       ��K	����A�*

logging/current_cost���;��+       ��K	����A�*

logging/current_costv��;ۍ�k+       ��K	��A�*

logging/current_costX��;�'�+       ��K	�9��A�*

logging/current_cost=��;I�./+       ��K	�f��A�*

logging/current_cost��;�7�+       ��K	����A�*

logging/current_cost���;^)��+       ��K	����A�*

logging/current_cost׈�;J�^%+       ��K	���A�*

logging/current_cost���;Ѷ�+       ��K	���A�*

logging/current_cost���;qR_+       ��K	=K��A�*

logging/current_coste��;!4*�+       ��K	@~��A�*

logging/current_cost?��;k?m�+       ��K	=���A�*

logging/current_cost|��;���+       ��K	9���A�*

logging/current_costL��;�|K�+       ��K	~ ��A�*

logging/current_cost䁉;�`� +       ��K	�6 ��A�*

logging/current_cost���;ᵡ�+       ��K	7f ��A�*

logging/current_cost���;f��+       ��K	W� ��A�*

logging/current_cost�~�;˸m+       ��K	�� ��A�*

logging/current_cost�~�; ���+       ��K	$� ��A�*

logging/current_cost�|�;��$y+       ��K	!!��A�*

logging/current_cost�z�;8T��+       ��K	rI!��A�*

logging/current_cost�y�;%��e+       ��K	�v!��A�*

logging/current_cost�x�;D��+       ��K	��!��A�*

logging/current_costx�;�`�i+       ��K	M�!��A�*

logging/current_cost�v�;���6+       ��K	N�!��A�*

logging/current_cost&v�;C���+       ��K	�."��A�*

logging/current_cost�t�;�&�O+       ��K	i\"��A�*

logging/current_cost=t�;��#Y+       ��K	�"��A�*

logging/current_cost s�;�r4�+       ��K	շ"��A�*

logging/current_costAr�;�6�v+       ��K	�"��A�*

logging/current_costkq�;Xf�-+       ��K	v#��A�*

logging/current_cost p�;f��0+       ��K	TA#��A�*

logging/current_cost�n�;���+       ��K	/m#��A�*

logging/current_cost4n�;��P�+       ��K	��#��A�*

logging/current_costm�;�ϙ+       ��K	X�#��A�*

logging/current_costTl�;��[+       ��K	��#��A�*

logging/current_costul�;�4i+       ��K	�#$��A�*

logging/current_cost&l�;ԇ��+       ��K	�R$��A�*

logging/current_costYi�;��)z+       ��K	π$��A�*

logging/current_cost�h�;�M�+       ��K	w�$��A�*

logging/current_cost�i�;�p�+       ��K	��$��A�*

logging/current_cost�g�;f<H+       ��K	V%��A�*

logging/current_costEf�;7�Q+       ��K	�4%��A�*

logging/current_cost�e�;���<+       ��K	�b%��A�*

logging/current_cost f�;��R.+       ��K		�%��A�*

logging/current_cost�d�;��Vc+       ��K	R�%��A�*

logging/current_cost�b�;���+       ��K	��%��A�*

logging/current_cost=b�;�s/�+       ��K	�&��A�*

logging/current_cost#a�;�+       ��K	�E&��A�*

logging/current_cost6`�;p�.�+       ��K	't&��A�*

logging/current_cost�_�;tQ�o+       ��K	n�&��A�*

logging/current_cost_�;��
�+       ��K	Y�&��A�*

logging/current_cost�^�;�;�Q+       ��K	�'��A�*

logging/current_costzd�; �o+       ��K	�<'��A�*

logging/current_cost9c�;�n+       ��K	po'��A�*

logging/current_costE\�;(�'!+       ��K	��'��A�*

logging/current_costi]�;�I�+       ��K	�'��A�*

logging/current_costG_�;���+       ��K	�(��A�*

logging/current_cost�Y�;��+       ��K	6(��A�*

logging/current_cost5Z�;��O+       ��K	ze(��A�*

logging/current_cost�X�;Kk{�+       ��K	(��A�*

logging/current_costX�;�1R+       ��K	L�(��A�*

logging/current_cost
W�;}��+       ��K	��(��A�*

logging/current_cost\W�;[���+       ��K	�/)��A�*

logging/current_costXV�;�
��+       ��K	�])��A�*

logging/current_cost�V�;�B�+       ��K	4�)��A�*

logging/current_cost�T�;�Ծ	+       ��K	�)��A�*

logging/current_costHU�;�+�+       ��K	a�)��A�*

logging/current_cost�S�;����+       ��K	I*��A�*

logging/current_costjS�;��y+       ��K	�L*��A�*

logging/current_costBS�;�Wf+       ��K	�|*��A�*

logging/current_cost~R�;��<�+       ��K	��*��A�*

logging/current_cost�Q�;+5��+       ��K	��*��A�*

logging/current_cost{Q�;�%��+       ��K	P	+��A�*

logging/current_cost�O�;,��^+       ��K	t<+��A�*

logging/current_cost�P�;��+       ��K	o+��A�*

logging/current_costLP�;���+       ��K	��+��A�*

logging/current_costO�;Q*p�+       ��K	I�+��A�*

logging/current_costCN�;�T��+       ��K	�+��A�*

logging/current_cost�M�;C{�+       ��K	�*,��A�*

logging/current_cost�M�;[��+       ��K	 Y,��A�*

logging/current_costJN�;{� +       ��K	"�,��A�*

logging/current_cost�K�;����+       ��K	��,��A�*

logging/current_costaN�;19�b+       ��K	��,��A�*

logging/current_costL�;�)��+       ��K	:-��A�*

logging/current_cost�J�;PW�+       ��K	K-��A�*

logging/current_cost�K�;��5)+       ��K	�~-��A�*

logging/current_costMJ�;���+       ��K	�-��A�*

logging/current_cost�I�;�TO+       ��K	��-��A�*

logging/current_cost_I�;�TB+       ��K	H.��A�*

logging/current_cost^H�;�v�+       ��K	�9.��A�*

logging/current_cost�G�;�0<]+       ��K	=h.��A�*

logging/current_cost�G�;�$�+       ��K	-�.��A�*

logging/current_cost1G�;�V�+       ��K	��.��A�*

logging/current_cost�F�;o�N+       ��K	��.��A�*

logging/current_costO@�;徏�+       ��K	�/��A�*

logging/current_cost):�;��5m+       ��K	K/��A�*

logging/current_costF2�;Τ��+       ��K	?y/��A�*

logging/current_cost�)�;�O�+       ��K	��/��A�*

logging/current_cost!�;n��A+       ��K	4�/��A�*

logging/current_costA�;� ��+       ��K	v0��A�*

logging/current_costb�;���+       ��K	�20��A�*

logging/current_cost��;�.}J+       ��K	�_0��A�*

logging/current_cost���;"k�+       ��K	 �0��A�*

logging/current_cost��;�n��+       ��K	��0��A�*

logging/current_costc�;��+       ��K	��0��A�*

logging/current_cost��;�	��+       ��K	�1��A�*

logging/current_cost/ۈ;��q+       ��K	H1��A�*

logging/current_cost�҈;
�K+       ��K	Qs1��A�*

logging/current_cost/ʈ;m�Z7+       ��K	��1��A�*

logging/current_cost���;5��Q+       ��K	�1��A�*

logging/current_cost\��;�R|+       ��K	0�1��A�*

logging/current_cost��;�N�+       ��K	�)2��A�*

logging/current_cost���;�:+       ��K	'Y2��A�*

logging/current_cost]��;viD�+       ��K	�2��A�*

logging/current_cost��;�.�]+       ��K	��2��A�*

logging/current_costϏ�;86g�+       ��K	��2��A�*

logging/current_cost���;�Y��+       ��K	�3��A�*

logging/current_costO�;0�+       ��K	L?3��A�*

logging/current_costw�;�P��+       ��K	?o3��A�*

logging/current_cost�n�;�j�~+       ��K	�3��A�*

logging/current_cost�g�;���h+       ��K	%�3��A�*

logging/current_cost�d�;`���+       ��K	4��A�*

logging/current_costdb�;��+       ��K	"34��A�*

logging/current_costI`�;�<U�+       ��K	b4��A�*

logging/current_costE]�;i��S+       ��K	��4��A�*

logging/current_costL[�;x!��+       ��K	��4��A�*

logging/current_cost�X�;�36�+       ��K	��4��A�*

logging/current_costiW�;�vǐ+       ��K	U5��A�*

logging/current_cost)W�;>�X�+       ��K	�H5��A�*

logging/current_cost`U�;�;AM+       ��K	�y5��A�*

logging/current_costQS�;i5!�+       ��K	��5��A�*

logging/current_cost�Q�;���`+       ��K	��5��A�*

logging/current_cost^R�;L� �+       ��K	36��A�*

logging/current_cost�P�;���=+       ��K	�36��A�*

logging/current_cost�N�;0?�F+       ��K	�d6��A�*

logging/current_cost'O�;Y�D+       ��K	��6��A�*

logging/current_costL�;\5TS+       ��K	��6��A�*

logging/current_costM�;Y:�'+       ��K	��6��A�*

logging/current_cost2K�;����+       ��K	h 7��A�*

logging/current_costJ�;��+       ��K	�P7��A�*

logging/current_costI�;tg��+       ��K	F�7��A�*

logging/current_cost�G�;J'�;+       ��K	�7��A�*

logging/current_cost�F�;�$�+       ��K	�7��A�*

logging/current_cost�E�;����+       ��K	8��A�*

logging/current_cost�E�;rzG�+       ��K	�E8��A�*

logging/current_costD�;�E+       ��K	�t8��A�*

logging/current_cost�B�;ۉ1Y+       ��K	u�8��A�*

logging/current_cost�A�;E�8�+       ��K	w�8��A�*

logging/current_cost%B�;���+       ��K	��8��A�*

logging/current_cost�?�;��&#+       ��K	�-9��A�*

logging/current_cost�@�;�-u+       ��K	^\9��A�*

logging/current_cost�?�;G�C@+       ��K	��9��A�*

logging/current_cost5=�;�}K�+       ��K	N�9��A�*

logging/current_costS=�;i3��+       ��K	E�9��A�*

logging/current_cost9=�;;T'+       ��K	�:��A�*

logging/current_cost�<�;���s+       ��K	N:��A�*

logging/current_cost�9�;���+       ��K	�{:��A�*

logging/current_cost�;�;j���+       ��K	��:��A�*

logging/current_cost�9�;����+       ��K	��:��A�*

logging/current_cost/8�;yb�&+       ��K	D;��A�*

logging/current_cost�6�;�"�+       ��K	=1;��A�*

logging/current_cost�6�;���+       ��K	O_;��A�*

logging/current_cost�6�;�l+       ��K	c�;��A�*

logging/current_cost�5�;�|�+       ��K	A�;��A�*

logging/current_costp5�;��Ш+       ��K	I<��A�*

logging/current_cost�4�;�b>+       ��K	5T<��A�*

logging/current_cost�2�;�@��+       ��K	�<��A�*

logging/current_cost�0�;?���+       ��K	��<��A�*

logging/current_cost-0�;�\�H+       ��K	F=��A�*

logging/current_coste/�;V��+       ��K	�9=��A�*

logging/current_cost9.�;��P-+       ��K	�|=��A�*

logging/current_cost3�;��	x+       ��K	C�=��A�*

logging/current_cost�/�;��s +       ��K	��=��A�*

logging/current_cost�,�;͘a+       ��K	v,>��A�*

logging/current_costD+�;J�^+       ��K	Xf>��A�*

logging/current_costq/�;+!+       ��K	��>��A�*

logging/current_cost�,�;pկ�+       ��K	'�>��A�*

logging/current_cost�(�;���+       ��K	A�>��A�*

logging/current_cost�*�;�u&+       ��K	�A?��A�*

logging/current_cost�*�;УD�+       ��K	��?��A�*

logging/current_cost�'�;�G'l+       ��K	�?��A�*

logging/current_cost�&�;佩b+       ��K	��?��A�*

logging/current_costa%�;��Q�+       ��K	@��A�*

logging/current_costC$�;o"�!+       ��K	�L@��A�*

logging/current_cost`$�;���+       ��K	�~@��A�*

logging/current_cost�"�;����+       ��K	L�@��A�*

logging/current_costU"�;KƯ8+       ��K	=�@��A�*

logging/current_cost7"�;����+       ��K	�A��A�*

logging/current_cost0$�;��{�+       ��K	EHA��A�*

logging/current_cost#�;�FZa+       ��K	�wA��A�*

logging/current_costS�;&�J�+       ��K	��A��A�*

logging/current_cost;�;h��0+       ��K	��A��A�*

logging/current_cost��;O�>�+       ��K	�B��A�*

logging/current_costF�;�v�+       ��K	)AB��A�*

logging/current_costt�;�R��+       ��K	$uB��A�*

logging/current_costy�;�� +       ��K	��B��A�*

logging/current_cost��;nh��+       ��K	W�B��A�*

logging/current_costD�;���+       ��K	�C��A�*

logging/current_costG�;�H��+       ��K	�9C��A�*

logging/current_costa�;��
�+       ��K	gC��A�*

logging/current_cost��;p+       ��K	ޕC��A�*

logging/current_costk�;7��+       ��K	\�C��A�*

logging/current_cost��;S�<�+       ��K	q�C��A�*

logging/current_cost��;﹓|+       ��K	�'D��A�*

logging/current_cost��;A4j;+       ��K	�VD��A�*

logging/current_cost��;����+       ��K	ȅD��A�*

logging/current_costY�;�@�+       ��K	
�D��A�*

logging/current_cost��;s���+       ��K	��D��A�*

logging/current_costR�;��6+       ��K	�E��A�*

logging/current_cost��;��+       ��K	�@E��A�*

logging/current_costN�;VQ$L+       ��K	
qE��A�*

logging/current_cost��;`q��+       ��K	��E��A�*

logging/current_cost��;X�^f+       ��K	4�E��A�*

logging/current_cost��;��� +       ��K	�F��A�*

logging/current_cost}�;8>�+       ��K	%1F��A�*

logging/current_cost]�;�dD�+       ��K	�_F��A�*

logging/current_cost��;_J�+       ��K	T�F��A�*

logging/current_cost�
�;ɭ��+       ��K	��F��A�*

logging/current_cost�	�;��m+       ��K	��F��A�*

logging/current_cost_�;���+       ��K	�G��A�*

logging/current_cost�;��+       ��K	dJG��A�*

logging/current_cost��;��H�+       ��K	�yG��A�*

logging/current_cost6�;�s}+       ��K	��G��A�*

logging/current_cost��;��+       ��K	S�G��A�*

logging/current_cost�;���b+       ��K	�H��A�*

logging/current_cost�;��"+       ��K	�0H��A�*

logging/current_cost �;�=X�+       ��K	�]H��A�*

logging/current_cost\�;Z�X�+       ��K	y�H��A�*

logging/current_cost��;B)��+       ��K	�H��A�*

logging/current_cost��;:���+       ��K	��H��A�*

logging/current_costz �;i�J+       ��K	� I��A�*

logging/current_cost� �;#��+       ��K	FQI��A�*

logging/current_costK��;7*	�+       ��K	>�I��A�*

logging/current_cost���;;l+       ��K	��I��A�*

logging/current_costL �;"x˹+       ��K	��I��A�*

logging/current_cost� �;�Pp+       ��K	�J��A�*

logging/current_cost���;�(��+       ��K	6;J��A�*

logging/current_cost=��;�튀+       ��K	�gJ��A�*

logging/current_cost���;Ç��+       ��K	ʕJ��A�*

logging/current_cost���;����+       ��K	~�J��A�*

logging/current_cost���;���O+       ��K	��J��A�*

logging/current_cost���;�V�^+       ��K	K��A�*

logging/current_costO��;�D��+       ��K	�LK��A�*

logging/current_cost���;m��>+       ��K	MyK��A�*

logging/current_cost��;��A@+       ��K	��K��A�*

logging/current_cost{��;�;�+       ��K	��K��A�*

logging/current_costs��;����+       ��K	0L��A�*

logging/current_cost���;1�K�+       ��K	�/L��A�*

logging/current_cost��;	,�'+       ��K	z]L��A�*

logging/current_costd��;|�n+       ��K	��L��A�*

logging/current_cost��;5���+       ��K	�L��A�*

logging/current_cost��;����+       ��K	��L��A�*

logging/current_cost1�;�aa�+       ��K	�M��A�*

logging/current_costE�;H�d+       ��K	�GM��A�*

logging/current_cost��;����+       ��K	�wM��A�*

logging/current_cost���;#��}+       ��K	��M��A�*

logging/current_costY��;~S�q+       ��K	��M��A�*

logging/current_cost�;��+       ��K	uN��A�*

logging/current_cost��;��+       ��K	�8N��A�*

logging/current_costl�;w�,+       ��K	#fN��A�*

logging/current_cost��;B)�S+       ��K	�N��A�*

logging/current_cost��;��Bf+       ��K	��N��A�*

logging/current_cost��;�3�+       ��K	7�N��A�*

logging/current_cost��;�{ +       ��K	O��A�*

logging/current_cost�
�;���+       ��K	
JO��A�*

logging/current_cost�;@�+       ��K	��O��A�*

logging/current_cost5�;HT��+       ��K	��O��A�*

logging/current_cost��;��.;+       ��K	h	P��A�*

logging/current_costc�;j��+       ��K	1PP��A�*

logging/current_cost*�;E�7+       ��K	ЏP��A�*

logging/current_cost��;%��+       ��K	��P��A�*

logging/current_cost��;d��+       ��K	�Q��A�*

logging/current_cost��;�t+       ��K	FQ��A�*

logging/current_cost�;�?�+       ��K		�Q��A�*

logging/current_cost��;�u�3+       ��K	d�Q��A�*

logging/current_cost��;&��+       ��K	O�Q��A�*

logging/current_cost��;���W+       ��K	;3R��A�*

logging/current_cost4�;�( �+       ��K	riR��A�*

logging/current_cost�݇;*��F+       ��K	��R��A�*

logging/current_cost�އ;*f{�+       ��K	��R��A�*

logging/current_cost.݇;�y��+       ��K	eS��A�*

logging/current_costo܇;���	+       ��K	�>S��A�*

logging/current_costۇ;d���+       ��K	�nS��A�*

logging/current_cost�އ;�
��+       ��K	5�S��A�*

logging/current_cost	ۇ;3�TJ+       ��K	�S��A�*

logging/current_cost�ه;<V�+       ��K	T��A�*

logging/current_cost�؇;j�7�+       ��K	�FT��A�*

logging/current_cost�ׇ;&���+       ��K	�uT��A�*

logging/current_cost�և;���+       ��K	��T��A�*

logging/current_costWׇ;��	+       ��K	��T��A�*

logging/current_cost{և;��s*+       ��K	cU��A�*

logging/current_cost�ԇ;+t�|+       ��K	=3U��A�*

logging/current_cost�Շ;f��	+       ��K	<aU��A�*

logging/current_cost
Շ;�K;+       ��K	��U��A�*

logging/current_cost�ԇ;��7+       ��K	ռU��A�*

logging/current_costBՇ;Z���+       ��K	2�U��A�*

logging/current_cost�Ї;�R�i+       ��K	2!V��A�*

logging/current_cost1ч;q�[k+       ��K	k\V��A�*

logging/current_cost�҇;����+       ��K	��V��A�*

logging/current_cost*҇;�`��+       ��K	2�V��A�*

logging/current_costv҇;��}+       ��K	^�V��A�*

logging/current_cost�͇;���3+       ��K	2W��A�*

logging/current_cost=·;k�o�+       ��K	�AW��A�*

logging/current_cost'͇;Y�-�+       ��K	$rW��A�*

logging/current_cost�ˇ;���+       ��K	��W��A�*

logging/current_costBˇ;��H�+       ��K	��W��A�*

logging/current_cost�̇;l��b+       ��K	��W��A�*

logging/current_cost�ˇ;}�Y�+       ��K	�)X��A�*

logging/current_cost�ˇ;��L+       ��K	yVX��A�*

logging/current_costv̇;[�G+       ��K	��X��A�*

logging/current_cost�ˇ;wm�}+       ��K	S�X��A�*

logging/current_cost	ɇ;i��4+       ��K	w�X��A�*

logging/current_costtȇ;yS+       ��K	�Y��A�*

logging/current_cost)Ǉ;���+       ��K	EY��A�*

logging/current_costkŇ;��ٌ+       ��K	�rY��A�*

logging/current_cost�Ƈ;Tg��+       ��K	��Y��A�*

logging/current_cost�ć;Qέl+       ��K	��Y��A�*

logging/current_cost�;D�[H+       ��K	�Y��A�*

logging/current_costpć;����+       ��K	�,Z��A�*

logging/current_costcÇ;,|�.+       ��K	�\Z��A�*

logging/current_costS��;��m�+       ��K	/�Z��A�*

logging/current_cost=ć;}��+       ��K	t�Z��A�*

logging/current_costH��;�W+       ��K	��Z��A�*

logging/current_cost=;�ҩq+       ��K	=[��A�*

logging/current_costT��;�J�+       ��K	�O[��A�*

logging/current_costо�;=.�+       ��K	��[��A�*

logging/current_costx��;��	6+       ��K	�[��A�*

logging/current_cost��;�^��+       ��K	o�[��A�*

logging/current_cost���;+A4K+       ��K	t\��A�*

logging/current_cost��;�<�w+       ��K	*:\��A�*

logging/current_cost���;���+       ��K	�q\��A�*

logging/current_cost���;���+       ��K	�\��A�*

logging/current_cost���;6��+       ��K	��\��A�*

logging/current_cost﹇;�s�+       ��K	��\��A�*

logging/current_cost׻�;%�U*+       ��K	50]��A�*

logging/current_cost���;��;+       ��K	�`]��A�*

logging/current_costC��;�I�+       ��K	L�]��A�*

logging/current_costD��;p�+       ��K	y�]��A�*

logging/current_cost���;�C��+       ��K	�]��A�*

logging/current_costN��;��iA+       ��K	8!^��A�*

logging/current_cost��;A}�&+       ��K	�O^��A�*

logging/current_cost���;I���+       ��K	�~^��A�*

logging/current_cost���;�0,+       ��K	e�^��A�*

logging/current_costȷ�;����+       ��K	 �^��A�*

logging/current_cost��;o���+       ��K	5_��A�*

logging/current_cost���;�P��+       ��K	9_��A�*

logging/current_cost鴇;��5L+       ��K	�n_��A�*

logging/current_cost���;�O�+       ��K	ҟ_��A�*

logging/current_cost]��;щ+       ��K	G�_��A�*

logging/current_cost*��;d�*l+       ��K	��_��A�*

logging/current_costF��;
�.J+       ��K	�)`��A�*

logging/current_costծ�; @d�+       ��K	=V`��A�*

logging/current_costӱ�;�)�+       ��K	.�`��A�*

logging/current_cost	��;K��}+       ��K	�`��A�*

logging/current_cost$��;C�m+       ��K	��`��A�*

logging/current_costo��;�O�+       ��K	�a��A�*

logging/current_cost���;�i[C+       ��K	yCa��A�*

logging/current_cost��;��E�+       ��K	pa��A�*

logging/current_costϩ�;�J�+       ��K	?�a��A�*

logging/current_costũ�;�s�F+       ��K	�a��A�*

logging/current_costۦ�;v/�7+       ��K	��a��A�*

logging/current_cost���;�:Ʈ+       ��K	�(b��A�*

logging/current_cost���;�HV+       ��K	;Vb��A�*

logging/current_cost���;��D$+       ��K	��b��A�*

logging/current_cost���;��+       ��K	1�b��A�*

logging/current_coste��;�n
D+       ��K	��b��A�*

logging/current_costĠ�;���+       ��K	�c��A�*

logging/current_cost9��;��K+       ��K	(;c��A�*

logging/current_costС�;�Ck�+       ��K	hc��A�*

logging/current_costd��;xŃ�+       ��K	7�c��A�*

logging/current_cost5��;6F�Z+       ��K	Y�c��A�*

logging/current_cost���;OEf+       ��K	��c��A�*

logging/current_cost7��;W,�+       ��K	�d��A�*

logging/current_costy��;2���+       ��K	&Od��A�*

logging/current_costꞇ;��UE+       ��K	9{d��A�*

logging/current_cost;��+       ��K	��d��A�*

logging/current_costj��;��8�+       ��K	6�d��A�*

logging/current_cost�;���P+       ��K	�e��A�*

logging/current_cost���;"��+       ��K	X/e��A�*

logging/current_cost1��;�mȘ+       ��K	�^e��A�*

logging/current_cost%��;1��+       ��K	�e��A�*

logging/current_cost)��;@fx�+       ��K	x�e��A�*

logging/current_cost�;���+       ��K	��e��A�*

logging/current_costϖ�;��=+       ��K	�f��A�*

logging/current_costە�;v���+       ��K	&Jf��A�*

logging/current_cost��;�Ֆz+       ��K	xf��A�*

logging/current_costÙ�;��֍+       ��K	;�f��A�*

logging/current_costn��;��b4+       ��K	��f��A�*

logging/current_cost���;k�N�+       ��K	�g��A�*

logging/current_costǦ�;,��j+       ��K	�4g��A�*

logging/current_cost<��;Y{_�+       ��K	�cg��A�*

logging/current_costD��;ɶ��+       ��K	?�g��A�*

logging/current_cost��;���+       ��K	W�g��A�*

logging/current_cost厇;��*+       ��K	��g��A�*

logging/current_cost���;�`+       ��K	�!h��A�*

logging/current_cost/��;��M+       ��K	PPh��A�*

logging/current_cost$��;�+       ��K	�~h��A�*

logging/current_cost��;00t{+       ��K	�h��A�*

logging/current_cost퍇;VѪ8+       ��K	g�h��A�*

logging/current_cost�;��,+       ��K	4
i��A�*

logging/current_cost���;,��+       ��K	 7i��A�*

logging/current_cost|��;��;5+       ��K	Zdi��A�*

logging/current_coste��;̈,�+       ��K	S�i��A� *

logging/current_costɋ�;�(+       ��K	��i��A� *

logging/current_cost눇;���V+       ��K	�i��A� *

logging/current_cost獇;ݣ(Q+       ��K	�%j��A� *

logging/current_cost'��;8� �+       ��K	�Zj��A� *

logging/current_costG��;���+       ��K	��j��A� *

logging/current_cost���;5=�k+       ��K	S�j��A� *

logging/current_cost���;�4<z+       ��K	��j��A� *

logging/current_costY��;��1+       ��K	�"k��A� *

logging/current_cost鈇;i�C�+       ��K	v]k��A� *

logging/current_cost��;�
�h+       ��K	ґk��A� *

logging/current_costu��;ew�+       ��K	��k��A� *

logging/current_cost)��;硁+       ��K	R�k��A� *

logging/current_cost��;$��+       ��K	5,l��A� *

logging/current_cost���;Bnw�+       ��K	al��A� *

logging/current_cost�~�;x(+       ��K	b�l��A� *

logging/current_costQ��;��3+       ��K	m�l��A� *

logging/current_cost�~�;e��+       ��K	��l��A� *

logging/current_costc~�;��+       ��K	b%m��A� *

logging/current_cost���;_���+       ��K	�Wm��A� *

logging/current_cost���;X��U+       ��K	�m��A� *

logging/current_cost|�;{�+       ��K	@�m��A� *

logging/current_costx��;�� �+       ��K	K�m��A� *

logging/current_cost�~�;d��+       ��K	Pn��A� *

logging/current_cost�{�;���e+       ��K	Gn��A� *

logging/current_costz�;3�+       ��K	�vn��A�!*

logging/current_cost���;�p�+       ��K	�n��A�!*

logging/current_cost��;�"�T+       ��K	P�n��A�!*

logging/current_cost�v�;�y�+       ��K	� o��A�!*

logging/current_costy�;�M2�+       ��K	f,o��A�!*

logging/current_coste|�;n>��+       ��K	�Yo��A�!*

logging/current_cost�x�;#�W+       ��K	��o��A�!*

logging/current_costy�;��u�+       ��K	�o��A�!*

logging/current_costnv�;�T�+       ��K	��o��A�!*

logging/current_cost$v�;0�+       ��K	6p��A�!*

logging/current_costMt�;��w�+       ��K	@p��A�!*

logging/current_costr�;��P�+       ��K	vlp��A�!*

logging/current_cost�r�;�h�+       ��K	2�p��A�!*

logging/current_cost�r�;h�r�+       ��K	��p��A�!*

logging/current_cost�t�;��l3+       ��K	�p��A�!*

logging/current_cost�s�;���0+       ��K	%q��A�!*

logging/current_cost�t�;�b ;+       ��K	�Qq��A�!*

logging/current_cost�p�;���M+       ��K	uq��A�!*

logging/current_cost:r�;���+       ��K	��q��A�!*

logging/current_costMn�;�l #+       ��K	��q��A�!*

logging/current_cost�l�;�0j+       ��K	V	r��A�!*

logging/current_cost~l�;�b�#+       ��K	W7r��A�!*

logging/current_cost�l�;���+       ��K	ir��A�!*

logging/current_cost{n�;���+       ��K	��r��A�!*

logging/current_cost�k�;Z`�+       ��K	�r��A�!*

logging/current_costDj�;���+       ��K	S�r��A�!*

logging/current_cost�h�;x�͛+       ��K	k s��A�"*

logging/current_cost�j�;m	�6+       ��K	�Ms��A�"*

logging/current_cost�i�;�y��+       ��K	�zs��A�"*

logging/current_cost�f�;�xA�+       ��K	e�s��A�"*

logging/current_cost"g�;�g"�+       ��K	�s��A�"*

logging/current_cost�e�;��I+       ��K	�t��A�"*

logging/current_cost�g�;���M+       ��K	�4t��A�"*

logging/current_cost�e�;��-�+       ��K	@ct��A�"*

logging/current_cost@d�;v.�?+       ��K	R�t��A�"*

logging/current_cost�e�;�c�+       ��K	��t��A�"*

logging/current_cost�f�;AY,+       ��K	q�t��A�"*

logging/current_cost�e�;��N�+       ��K	�"u��A�"*

logging/current_cost�a�;9E�+       ��K	�Qu��A�"*

logging/current_costQf�;
�O+       ��K	L~u��A�"*

logging/current_cost�e�; �d�+       ��K	ѫu��A�"*

logging/current_cost�a�;��i+       ��K	`�u��A�"*

logging/current_cost�`�;5��J+       ��K	-v��A�"*

logging/current_cost$c�;#H^2+       ��K	2v��A�"*

logging/current_cost"d�;]d\++       ��K	�`v��A�"*

logging/current_costO\�;��Ui+       ��K	��v��A�"*

logging/current_cost_�;�9��+       ��K	�v��A�"*

logging/current_cost�^�;�>I+       ��K	��v��A�"*

logging/current_cost�Z�;,�m�+       ��K	�w��A�"*

logging/current_cost][�;*>�J+       ��K	�Kw��A�"*

logging/current_cost�Y�;4M+       ��K	�{w��A�"*

logging/current_cost�[�;F�R6+       ��K	��w��A�#*

logging/current_cost�Y�;���+       ��K	i�w��A�#*

logging/current_costX�;�Ƥ�+       ��K	Jx��A�#*

logging/current_cost�[�;���+       ��K	R5x��A�#*

logging/current_costv\�;�4m+       ��K	�`x��A�#*

logging/current_cost@Y�;7+       ��K	ѐx��A�#*

logging/current_cost�V�;�n+       ��K	?�x��A�#*

logging/current_cost�]�;�0{�+       ��K	�x��A�#*

logging/current_costtS�;Ui�+       ��K	�y��A�#*

logging/current_cost�U�;K�z�+       ��K	�Ky��A�#*

logging/current_costW�;���n+       ��K	�|y��A�#*

logging/current_cost�T�;���)+       ��K	�y��A�#*

logging/current_costnQ�;5x��+       ��K	'�y��A�#*

logging/current_costLS�;�J<�+       ��K	�z��A�#*

logging/current_cost4S�;�`6+       ��K	�4z��A�#*

logging/current_cost�S�;�?=+       ��K	{cz��A�#*

logging/current_cost�P�;N�Y+       ��K	�z��A�#*

logging/current_cost�N�;)��+       ��K	^�z��A�#*

logging/current_cost P�;�X^Z+       ��K	��z��A�#*

logging/current_costN�;@C�!+       ��K	�{��A�#*

logging/current_costvN�;����+       ��K	
J{��A�#*

logging/current_costN�;�A��+       ��K	s�{��A�#*

logging/current_cost�M�;��+       ��K	��{��A�#*

logging/current_cost�O�;M2"k+       ��K	<	|��A�#*

logging/current_cost�M�;�:��+       ��K	�=|��A�#*

logging/current_cost=O�;K:$�+       ��K	��|��A�#*

logging/current_cost�K�;�v�F+       ��K	g�|��A�$*

logging/current_costsG�;96JI+       ��K	U�|��A�$*

logging/current_cost�I�;���+       ��K	�&}��A�$*

logging/current_costKG�;)#0�+       ��K	0d}��A�$*

logging/current_cost`E�;�vI+       ��K	��}��A�$*

logging/current_cost�E�;k���+       ��K	O�}��A�$*

logging/current_costmG�;�Y�+       ��K	a~��A�$*

logging/current_costOL�;03��+       ��K	�P~��A�$*

logging/current_costD�;�,(h+       ��K	��~��A�$*

logging/current_costoK�;��D+       ��K		�~��A�$*

logging/current_cost\C�;%M?l+       ��K	��~��A�$*

logging/current_costnH�;g��+       ��K	f'��A�$*

logging/current_cost�E�;�]�+       ��K	d��A�$*

logging/current_costuE�;��K+       ��K	O���A�$*

logging/current_cost�F�;�a�+       ��K	 ���A�$*

logging/current_cost�A�;�ܼo+       ��K	���A�$*

logging/current_cost?E�;��K}+       ��K	b=���A�$*

logging/current_cost@=�;(�z+       ��K	�p���A�$*

logging/current_cost�@�;��
1+       ��K	`����A�$*

logging/current_cost!=�;-�+       ��K	�Ѐ��A�$*

logging/current_cost�>�;[�i-+       ��K	���A�$*

logging/current_cost�:�;��+       ��K	�/���A�$*

logging/current_cost�<�;���+       ��K	_^���A�$*

logging/current_cost�9�;]��+       ��K	|����A�$*

logging/current_cost�9�;"�me+       ��K	p����A�$*

logging/current_costLJ�;,�e�+       ��K	ꁞ�A�$*

logging/current_cost)8�;�h�+       ��K	Y ���A�%*

logging/current_costN9�;6���+       ��K	R���A�%*

logging/current_cost#9�;�̜�+       ��K	؆���A�%*

logging/current_cost\7�;!�(^+       ��K	:����A�%*

logging/current_cost8�;#��+       ��K	�ゞ�A�%*

logging/current_cost45�;|H�l+       ��K	����A�%*

logging/current_cost�6�;���d+       ��K	A���A�%*

logging/current_cost�9�;a��l+       ��K	p���A�%*

logging/current_cost�4�;4JyS+       ��K	�����A�%*

logging/current_costx1�;�SL�+       ��K	�σ��A�%*

logging/current_cost13�;��4y+       ��K	t����A�%*

logging/current_cost(3�;5j#�+       ��K	�-���A�%*

logging/current_cost55�; ��+       ��K	Ka���A�%*

logging/current_cost-8�;=�b_+       ��K	ˑ���A�%*

logging/current_costS4�;\T"+       ��K	m��A�%*

logging/current_cost:1�;�š�+       ��K	N��A�%*

logging/current_cost/�;��(�+       ��K	8"���A�%*

logging/current_cost�-�;1��}+       ��K	�Q���A�%*

logging/current_cost�0�;�VH>+       ��K	���A�%*

logging/current_cost[-�;(+��+       ��K	����A�%*

logging/current_cost3/�;�tƒ+       ��K	�����A�%*

logging/current_cost�2�;��>�+       ��K	����A�%*

logging/current_costm)�;��+       ��K	>���A�%*

logging/current_cost�)�;_9ac+       ��K	�q���A�%*

logging/current_costc)�;��+       ��K	v����A�%*

logging/current_cost�5�;�f��+       ��K	�Ά��A�&*

logging/current_costq-�;Iz��+       ��K	>����A�&*

logging/current_costn'�;c	�0+       ��K	�-���A�&*

logging/current_cost�&�;嫰+       ��K	Z[���A�&*

logging/current_cost,�;?�m�+       ��K	�����A�&*

logging/current_cost*�;q0��+       ��K	�����A�&*

logging/current_cost�'�;
U�5+       ��K	臞�A�&*

logging/current_costZ#�;���+       ��K	H���A�&*

logging/current_cost^"�;�r�q+       ��K	�C���A�&*

logging/current_cost�(�;�U�+       ��K	p���A�&*

logging/current_cost�$�;�h�`+       ��K	����A�&*

logging/current_cost%�;F�+       ��K	<Ј��A�&*

logging/current_costs*�;3P��+       ��K	�����A�&*

logging/current_cost��;d���+       ��K	m?���A�&*

logging/current_cost>'�;�:��+       ��K	�����A�&*

logging/current_costc �;%��+       ��K	=����A�&*

logging/current_cost0 �;Y�]�+       ��K	����A�&*

logging/current_cost+$�;�+b�+       ��K	�.���A�&*

logging/current_cost,�;N���+       ��K	�a���A�&*

logging/current_costQ$�;B��+       ��K	{����A�&*

logging/current_costd�;���+       ��K	�͊��A�&*

logging/current_costL�;��/-+       ��K	���A�&*

logging/current_cost3�;�3"#+       ��K	@���A�&*

logging/current_costD�;�U�+       ��K	�w���A�&*

logging/current_cost��;��+       ��K	�����A�&*

logging/current_cost�;���+       ��K	�䋞�A�&*

logging/current_cost��;�U��+       ��K	����A�'*

logging/current_costr�;��!y+       ��K	�M���A�'*

logging/current_cost��;F�v�+       ��K	‌��A�'*

logging/current_cost��;��| +       ��K	����A�'*

logging/current_costy�;�곕+       ��K	�ތ��A�'*

logging/current_cost��;>m��+       ��K	����A�'*

logging/current_cost��;1D�+       ��K	D<���A�'*

logging/current_costd�;���H+       ��K	�o���A�'*

logging/current_cost}�;v�
+       ��K	����A�'*

logging/current_cost��;�
+       ��K	�΍��A�'*

logging/current_cost��;�� +       ��K	�����A�'*

logging/current_cost��;W�Q�+       ��K	m*���A�'*

logging/current_cost��;(%+       ��K	�W���A�'*

logging/current_cost��;fF��+       ��K	P����A�'*

logging/current_cost�;`2�|+       ��K	�����A�'*

logging/current_costf�;��+       ��K	z莞�A�'*

logging/current_cost��;���+       ��K	���A�'*

logging/current_costQ�;5�0�+       ��K	]G���A�'*

logging/current_cost��;\/�+       ��K	�t���A�'*

logging/current_cost��;��^�+       ��K	�����A�'*

logging/current_costB�;��b+       ��K	FᏞ�A�'*

logging/current_cost��;�W�/+       ��K	����A�'*

logging/current_costj�;�sJ+       ��K	E���A�'*

logging/current_cost��;*�Q�+       ��K	�u���A�'*

logging/current_cost��;�f�q+       ��K	z����A�'*

logging/current_cost��;Ycm�+       ��K	�Ӑ��A�(*

logging/current_costT	�;��+       ��K	����A�(*

logging/current_cost2	�;��H�+       ��K	�4���A�(*

logging/current_costF	�;Y�+       ��K	 f���A�(*

logging/current_cost��;��3j+       ��K	x����A�(*

logging/current_cost��;���+       ��K	ʑ��A�(*

logging/current_cost�	�;Ef�+       ��K	�����A�(*

logging/current_costw	�;�Ʃ�+       ��K	8&���A�(*

logging/current_cost��;���s+       ��K	�S���A�(*

logging/current_cost��;%?�+       ��K	Ƀ���A�(*

logging/current_cost&�;&A+       ��K	�����A�(*

logging/current_cost��;{GO+       ��K	}撞�A�(*

logging/current_costq �;���7+       ��K	����A�(*

logging/current_cost��;w���+       ��K	)K���A�(*

logging/current_cost< �;��y+       ��K	�y���A�(*

logging/current_cost�;�x��+       ��K	����A�(*

logging/current_costM��;�e5�+       ��K	�ޓ��A�(*

logging/current_cost!�;�w�+       ��K	����A�(*

logging/current_cost���;�=%y+       ��K	a;���A�(*

logging/current_cost� �;0g�m+       ��K	oi���A�(*

logging/current_cost=��;��+       ��K	�����A�(*

logging/current_cost���;�</+       ��K	,ʔ��A�(*

logging/current_cost���;��1�+       ��K	����A�(*

logging/current_cost ��;�a�+       ��K	:'���A�(*

logging/current_cost���;xF�x+       ��K	�U���A�(*

logging/current_cost���;Lz�X+       ��K	����A�(*

logging/current_costI��;����+       ��K	q����A�)*

logging/current_cost}��;�|$�+       ��K	�ߕ��A�)*

logging/current_costy��;�C9[+       ��K	����A�)*

logging/current_cost��;�q�+       ��K	�G���A�)*

logging/current_cost\��;�_$�+       ��K	av���A�)*

logging/current_cost>��;z�+       ��K	����A�)*

logging/current_cost���;�.}8+       ��K	�Ӗ��A�)*

logging/current_cost���;�d�+       ��K	���A�)*

logging/current_cost��;�&�+       ��K	=5���A�)*

logging/current_cost��;��e+       ��K	Vd���A�)*

logging/current_cost���;�7��+       ��K	r����A�)*

logging/current_cost
��;.�N�+       ��K	Ϳ���A�)*

logging/current_cost��;Z*u�+       ��K	�A�)*

logging/current_cost��;x|+       ��K	1���A�)*

logging/current_cost���;
�v,+       ��K	�J���A�)*

logging/current_cost!��;9 +       ��K	/x���A�)*

logging/current_cost��;F㭴+       ��K	ޤ���A�)*

logging/current_costb��;�ݫ�+       ��K	�Ԙ��A�)*

logging/current_cost�;D�0�+       ��K	����A�)*

logging/current_cost��;&�OR+       ��K	�3���A�)*

logging/current_cost���;���$+       ��K	�b���A�)*

logging/current_cost��;DC+       ��K	����A�)*

logging/current_costB�;�"~�+       ��K	�����A�)*

logging/current_costc�;�t��+       ��K	=뙞�A�)*

logging/current_cost��;�:_+       ��K	����A�)*

logging/current_cost
�;��S�+       ��K	HF���A�)*

logging/current_cost��;5T��+       ��K	Ju���A�**

logging/current_cost��;�6�O+       ��K	%����A�**

logging/current_cost1�;7��J+       ��K	jҚ��A�**

logging/current_cost�;�~_+       ��K	� ���A�**

logging/current_costB�;�19#+       ��K	f.���A�**

logging/current_cost��;��Rn+       ��K	[���A�**

logging/current_cost?�;^d�m+       ��K	ɇ���A�**

logging/current_cost��;��+       ��K	l����A�**

logging/current_costg�;[ZB3+       ��K	u��A�**

logging/current_cost��;H�u+       ��K	$!���A�**

logging/current_cost��;��z�+       ��K	,P���A�**

logging/current_cost��;�|_�+       ��K	�}���A�**

logging/current_cost��;#\��+       ��K	s����A�**

logging/current_cost��;k���+       ��K	.ڜ��A�**

logging/current_cost��;�o�V+       ��K	�	���A�**

logging/current_costB�;C��+       ��K	�4���A�**

logging/current_cost�ކ;��<+       ��K	�d���A�**

logging/current_costm�;�D�+       ��K	K����A�**

logging/current_cost�ކ;���+       ��K	~����A�**

logging/current_cost��;	��+       ��K		��A�**

logging/current_cost�܆;.vpT+       ��K	�"���A�**

logging/current_cost��;��D+       ��K	|Q���A�**

logging/current_cost�߆;�fMc+       ��K	���A�**

logging/current_cost�܆;
�?+       ��K	g����A�**

logging/current_cost�݆;I��+       ��K	�����A�**

logging/current_costc݆;���+       ��K	���A�+*

logging/current_cost�ن;R+�+       ��K	�:���A�+*

logging/current_cost)ن;Aa4�+       ��K	"i���A�+*

logging/current_costvن;�Q�+       ��K	蚟��A�+*

logging/current_cost�ކ;�+       ��K	*ȟ��A�+*

logging/current_costh؆;H�Η+       ��K	�����A�+*

logging/current_cost��;��;+       ��K	�%���A�+*

logging/current_cost�Ԇ;�8�+       ��K	�U���A�+*

logging/current_costUۆ;����+       ��K	ȃ���A�+*

logging/current_cost;ن;(%+       ��K	�����A�+*

logging/current_cost�ن;��Ɖ+       ��K	�ࠞ�A�+*

logging/current_cost�ӆ;�+�+       ��K	����A�+*

logging/current_costIֆ;��y+       ��K	%=���A�+*

logging/current_cost҆;�R�+       ��K	�i���A�+*

logging/current_cost#҆;�$g+       ��K	ǚ���A�+*

logging/current_costц;D���+       ��K	ˡ��A�+*

logging/current_cost?ӆ;��m+       ��K	>����A�+*

logging/current_costnֆ;�L��+       ��K	5(���A�+*

logging/current_costuІ;�`�+       ��K	
U���A�+*

logging/current_cost�׆;��0U+       ��K	g����A�+*

logging/current_costl̆;���V+       ��K	�����A�+*

logging/current_cost�Ն;t@�+       ��K	�ݢ��A�+*

logging/current_cost{ц;&�+�+       ��K	����A�+*

logging/current_cost@Ն;�q w+       ��K	�9���A�+*

logging/current_cost͆;��+       ��K	Re���A�+*

logging/current_cost?͆;@��+       ��K	 ����A�+*

logging/current_cost�ˆ;j�Ң+       ��K	����A�,*

logging/current_costPʆ;�F��+       ��K	2�A�,*

logging/current_cost�ʆ;�u�+       ��K	#���A�,*

logging/current_costbˆ;o�A�+       ��K	�J���A�,*

logging/current_cost�ˆ;/5]_+       ��K	z���A�,*

logging/current_costɆ;�_+       ��K	ԧ���A�,*

logging/current_cost�ʆ;�Pcf+       ��K	^֤��A�,*

logging/current_costf̆;XeL�+       ��K	����A�,*

logging/current_costUφ;��z�+       ��K	K2���A�,*

logging/current_cost�Ȇ; ݯN+       ��K	�`���A�,*

logging/current_cost
ǆ;�;F�+       ��K	U����A�,*

logging/current_costnΆ;;z�+       ��K	ι���A�,*

logging/current_cost�Ć;C��$+       ��K	�神�A�,*

logging/current_cost/Ɔ;�Mb+       ��K	���A�,*

logging/current_costƆ;�L��+       ��K	�E���A�,*

logging/current_cost�Æ;�|�+       ��K	�u���A�,*

logging/current_cost�Ɔ;M�a+       ��K	�����A�,*

logging/current_cost_ǆ;��g8+       ��K	@Ӧ��A�,*

logging/current_cost�Ɔ;���+       ��K	����A�,*

logging/current_cost5ʆ;#���+       ��K	�4���A�,*

logging/current_costm��;�A+       ��K	b���A�,*

logging/current_costi��;��fB+       ��K	����A�,*

logging/current_cost���;�1+       ��K	X����A�,*

logging/current_cost���;�qx+       ��K	駞�A�,*

logging/current_costZ��;1h�O+       ��K	>���A�,*

logging/current_cost;qNZ�+       ��K	wE���A�-*

logging/current_cost-��;�-ڜ+       ��K	�u���A�-*

logging/current_cost���;k�V�+       ��K	N����A�-*

logging/current_cost�ʆ;&���+       ��K	�٨��A�-*

logging/current_cost��;��X$+       ��K	t���A�-*

logging/current_cost�ǆ;�Y�z+       ��K	X5���A�-*

logging/current_costJ��;zN8+       ��K	�a���A�-*

logging/current_cost-��;��N+       ��K	����A�-*

logging/current_cost5��;It�+       ��K	�ĩ��A�-*

logging/current_cost1��;��"�+       ��K	���A�-*

logging/current_cost��;�8`�+       ��K	�!���A�-*

logging/current_cost>��;�L�"+       ��K	R���A�-*

logging/current_cost
��;/�m�+       ��K	�����A�-*

logging/current_cost㲆;�2�T+       ��K	2爵�A�-*

logging/current_cost���;Tz��+       ��K	����A�-*

logging/current_cost��;d�l+       ��K	�P���A�-*

logging/current_cost���;�}4+       ��K	ڄ���A�-*

logging/current_cost���;����+       ��K	۶���A�-*

logging/current_cost��;wK�+       ��K	�嫞�A�-*

logging/current_cost���;ʛ�~+       ��K	@���A�-*

logging/current_cost��;
D�+       ��K	�M���A�-*

logging/current_costή�;u?��+       ��K	����A�-*

logging/current_cost���;8 ��+       ��K	�����A�-*

logging/current_cost��;��X+       ��K	߬��A�-*

logging/current_cost���;��[+       ��K	f���A�-*

logging/current_costS��;�)5+       ��K	�>���A�-*

logging/current_costܮ�;Zm�+       ��K	Pm���A�.*

logging/current_costq��;3<��+       ��K	�����A�.*

logging/current_costR��;F3��+       ��K	�ѭ��A�.*

logging/current_costn��;��Q+       ��K	�����A�.*

logging/current_costC��;�+       ��K	�+���A�.*

logging/current_cost~��;��+       ��K	�[���A�.*

logging/current_cost��;�x�+       ��K	͊���A�.*

logging/current_cost���;�)H+       ��K	5����A�.*

logging/current_cost^��;e�+       ��K	�讞�A�.*

logging/current_cost"��;͟k+       ��K	����A�.*

logging/current_cost���;N�d�+       ��K	E���A�.*

logging/current_cost+��;��T�+       ��K	�p���A�.*

logging/current_costפ�;jW�+       ��K	󟯞�A�.*

logging/current_cost-��;���+       ��K	�ϯ��A�.*

logging/current_cost��;q���+       ��K	�����A�.*

logging/current_cost���;_y��+       ��K	�+���A�.*

logging/current_cost\��;�0+       ��K	$Y���A�.*

logging/current_costΤ�;.��T+       ��K	�����A�.*

logging/current_costZ��;�zl+       ��K	����A�.*

logging/current_costϡ�;�Ѝ+       ��K	�ᰞ�A�.*

logging/current_costܣ�;a|,i+       ��K	s���A�.*

logging/current_costI��;�0*+       ��K	M@���A�.*

logging/current_cost���;WC &+       ��K	:q���A�.*

logging/current_cost;F{8+       ��K	x����A�.*

logging/current_costß�;�~�m+       ��K	�˱��A�.*

logging/current_costߜ�;S�*+       ��K	g����A�.*

logging/current_costK��;��[+       ��K	'���A�/*

logging/current_costI��;�+%�+       ��K	�U���A�/*

logging/current_cost���;ޛT�+       ��K	A����A�/*

logging/current_cost~��;�AQ\+       ��K	C����A�/*

logging/current_cost]��;_%�;+       ��K	�Პ�A�/*

logging/current_cost˙�;s�T�+       ��K	����A�/*

logging/current_costř�;��5(+       ��K	�=���A�/*

logging/current_coste��;���+       ��K	Ok���A�/*

logging/current_costq��;ɰ1�+       ��K	L����A�/*

logging/current_cost蛆;ǰu�+       ��K	ȳ��A�/*

logging/current_cost'��;Q�i�+       ��K	F����A�/*

logging/current_cost;��;��6+       ��K	F ���A�/*

logging/current_cost���;I�@+       ��K	�L���A�/*

logging/current_cost(��;iIE�+       ��K	sz���A�/*

logging/current_cost���;���+       ��K	�����A�/*

logging/current_costږ�;��+       ��K	�ִ��A�/*

logging/current_cost��;��d+       ��K	����A�/*

logging/current_cost>��;k�&�+       ��K	i2���A�/*

logging/current_costv��;�/��+       ��K	Qa���A�/*

logging/current_cost���;��+       ��K	�����A�/*

logging/current_cost���;a<�+       ��K	����A�/*

logging/current_cost~��;�3�e+       ��K	'뵞�A�/*

logging/current_cost���;��7+       ��K	����A�/*

logging/current_costݓ�;�b�Q+       ��K	^G���A�/*

logging/current_costW��;y4��+       ��K	Zv���A�/*

logging/current_costX��;��0+       ��K	�����A�0*

logging/current_cost��;�
3�+       ��K	�Ӷ��A�0*

logging/current_costN��;Q���+       ��K	����A�0*

logging/current_cost���;)��q+       ��K	�1���A�0*

logging/current_costj��;�#�+       ��K	�b���A�0*

logging/current_cost���;і+       ��K	�����A�0*

logging/current_cost���;�k�#+       ��K	�����A�0*

logging/current_cost���;���/+       ��K	^�A�0*

logging/current_cost���;%��+       ��K	j���A�0*

logging/current_cost���;�T+       ��K	�N���A�0*

logging/current_cost؋�;t_^�+       ��K	}���A�0*

logging/current_cost���;1$�q+       ��K	�����A�0*

logging/current_cost_��;�3q+       ��K	ڸ��A�0*

logging/current_costz��;��N�+       ��K	V���A�0*

logging/current_costW��;��
�+       ��K	i7���A�0*

logging/current_costՅ�;�� �+       ��K	�c���A�0*

logging/current_costo��;$�+       ��K	����A�0*

logging/current_costۇ�;�Ï�+       ��K	�ù��A�0*

logging/current_cost-��;��d�+       ��K	���A�0*

logging/current_cost���;���+       ��K	� ���A�0*

logging/current_cost���;��e�+       ��K	�O���A�0*

logging/current_cost���;a��+       ��K	:���A�0*

logging/current_costn��;�� �+       ��K	�����A�0*

logging/current_cost?��;@Lw~+       ��K	�ۺ��A�0*

logging/current_cost���;?��+       ��K	����A�0*

logging/current_cost.��;Q��+       ��K	/;���A�0*

logging/current_cost��;����+       ��K	�h���A�1*

logging/current_costq~�;863�+       ��K	/����A�1*

logging/current_costC~�;%�q�+       ��K	����A�1*

logging/current_cost(��;=��+       ��K	]6���A�1*

logging/current_cost8}�;���+       ��K	p���A�1*

logging/current_cost�|�;?��\+       ��K	�����A�1*

logging/current_costiz�;&8�+       ��K	+鼞�A�1*

logging/current_cost�;�� +       ��K	A!���A�1*

logging/current_cost!��;��p+       ��K	^���A�1*

logging/current_cost]��;]��@+       ��K	�����A�1*

logging/current_cost�-�; ,UM+       ��K	6ҽ��A�1*

logging/current_cost�,�;D�+       ��K	-���A�1*

logging/current_cost�=�;ǯA+       ��K	G���A�1*

logging/current_cost�؂;��R+       ��K	|����A�1*

logging/current_cost��;��8�+       ��K	�����A�1*

logging/current_cost�Ȃ;V�+       ��K	�����A�1*

logging/current_costz��;�)r�+       ��K	�7���A�1*

logging/current_cost���;��+       ��K	����A�1*

logging/current_cost���;�R+       ��K	�˿��A�1*

logging/current_costR��;���)+       ��K	e���A�1*

logging/current_cost���;X��+       ��K	;���A�1*

logging/current_cost���;�)+       ��K	Wk���A�1*

logging/current_cost ��;}|+       ��K	����A�1*

logging/current_cost��;�v��+       ��K	�����A�1*

logging/current_cost��;Pq�'+       ��K	p���A�1*

logging/current_cost���;:k�C+       ��K	�:���A�2*

logging/current_cost���;zs�+       ��K	�l���A�2*

logging/current_costƟ�;�A�+       ��K	7����A�2*

logging/current_cost���;�ݾ+       ��K	3����A�2*

logging/current_cost���;���+       ��K	o�A�2*

logging/current_cost��;�L�+       ��K	�D�A�2*

logging/current_costá�;^�7+       ��K	�{�A�2*

logging/current_cost���;�)�+       ��K	���A�2*

logging/current_costҜ�;�;�+       ��K	>TÞ�A�2*

logging/current_cost�;G}�z+       ��K	�Þ�A�2*

logging/current_costx��;�+       ��K	l�Þ�A�2*

logging/current_cost)��;��g�+       ��K	[Ğ�A�2*

logging/current_cost���;qqȽ+       ��K	�\Ğ�A�2*

logging/current_cost���;��:+       ��K	.�Ğ�A�2*

logging/current_cost
��;e��+       ��K	m�Ğ�A�2*

logging/current_cost��;�.E+       ��K	�Ş�A�2*

logging/current_cost	��;�.r�+       ��K	�fŞ�A�2*

logging/current_cost���;5�Y+       ��K	�Ş�A�2*

logging/current_costF��;��9|+       ��K	��Ş�A�2*

logging/current_cost���;�,Ό+       ��K	�ƞ�A�2*

logging/current_cost��;{�>F+       ��K	�Rƞ�A�2*

logging/current_cost/��;(�K+       ��K	P�ƞ�A�2*

logging/current_cost���;*mx+       ��K	J�ƞ�A�2*

logging/current_cost̒�;�F�+       ��K	�ƞ�A�2*

logging/current_cost��;�ׯ/+       ��K	�/Ǟ�A�2*

logging/current_cost���;*���+       ��K	�fǞ�A�2*

logging/current_costh��;V��P+       ��K	�Ǟ�A�3*

logging/current_costc��;ߠ�e+       ��K	�Ǟ�A�3*

logging/current_costK��;�2�+       ��K	WȞ�A�3*

logging/current_cost`��;�˙+       ��K	u=Ȟ�A�3*

logging/current_cost���;��}g+       ��K	f�Ȟ�A�3*

logging/current_cost���;�]��+       ��K	��Ȟ�A�3*

logging/current_cost�;0cz+       ��K	ɞ�A�3*

logging/current_cost�;SP�<+       ��K	�Mɞ�A�3*

logging/current_costC��;!��+       ��K	&�ɞ�A�3*

logging/current_cost|��;��9�+       ��K	-�ɞ�A�3*

logging/current_cost���;�Q(+       ��K	��ɞ�A�3*

logging/current_cost&��;`ݧk+       ��K	x5ʞ�A�3*

logging/current_cost���;��jv+       ��K	�lʞ�A�3*

logging/current_cost,��;Vy��+       ��K	��ʞ�A�3*

logging/current_cost���;\HTE+       ��K	��ʞ�A�3*

logging/current_cost2��;(��t+       ��K	�	˞�A�3*

logging/current_cost��;���+       ��K	7E˞�A�3*

logging/current_costՇ�;ɶ�+       ��K	Bx˞�A�3*

logging/current_costh��;?ô�+       ��K	"�˞�A�3*

logging/current_costQ��;�+       ��K	N�˞�A�3*

logging/current_cost<��;ηY�+       ��K	(̞�A�3*

logging/current_cost��;�A�+       ��K	8@̞�A�3*

logging/current_cost̄�;��p+       ��K	
u̞�A�3*

logging/current_cost]��;�k��+       ��K	�̞�A�3*

logging/current_cost˃�;9�f+       ��K	��̞�A�3*

logging/current_costG��;@��+       ��K	�͞�A�3*

logging/current_cost���;�(�+       ��K	�?͞�A�4*

logging/current_costv��;Ģ�+       ��K	�͞�A�4*

logging/current_costD��;7��.+       ��K	��͞�A�4*

logging/current_costz��;x^^+       ��K	-�͞�A�4*

logging/current_cost�;��:�+       ��K	Ξ�A�4*

logging/current_cost适;�׭E+       ��K	KΞ�A�4*

logging/current_cost5��;�+       ��K	aΞ�A�4*

logging/current_costP|�;�݋+       ��K	�Ξ�A�4*

logging/current_cost=}�;S�4+       ��K	��Ξ�A�4*

logging/current_cost�}�;���9+       ��K	�Ϟ�A�4*

logging/current_cost�y�;w�)+       ��K	�=Ϟ�A�4*

logging/current_cost���;��p+       ��K	�mϞ�A�4*

logging/current_costz��;���+       ��K	��Ϟ�A�4*

logging/current_cost�|�;Y	��+       ��K	��Ϟ�A�4*

logging/current_cost|�;���+       ��K	!О�A�4*

logging/current_cost�}�;l	+       ��K	�QО�A�4*

logging/current_costw�;�+a+       ��K	�О�A�4*

logging/current_costbx�;y� �+       ��K	g�О�A�4*

logging/current_cost_v�;^���+       ��K	r�О�A�4*

logging/current_cost�t�;�tL+       ��K	�#ў�A�4*

logging/current_costlw�;�?jA+       ��K	 Sў�A�4*

logging/current_costBx�;�!(�+       ��K	��ў�A�4*

logging/current_cost�~�;+�?u+       ��K	%�ў�A�4*

logging/current_cost�q�;�+Ŕ+       ��K	t�ў�A�4*

logging/current_costd|�;��t+       ��K	WҞ�A�4*

logging/current_cost�p�;�57 +       ��K	�LҞ�A�5*

logging/current_cost�q�;��+       ��K	}Ҟ�A�5*

logging/current_costfv�;��T+       ��K	K�Ҟ�A�5*

logging/current_costUo�;��]*+       ��K	��Ҟ�A�5*

logging/current_cost�p�;h��+       ��K	�Ӟ�A�5*

logging/current_costzo�;9��+       ��K	d?Ӟ�A�5*

logging/current_costs�;�|�+       ��K	�mӞ�A�5*

logging/current_cost�p�;���+       ��K	��Ӟ�A�5*

logging/current_cost6o�;�Pq+       ��K	�Ӟ�A�5*

logging/current_cost�q�;_��m+       ��K	�Ԟ�A�5*

logging/current_cost k�;x�@+       ��K	�2Ԟ�A�5*

logging/current_cost	r�;8]��+       ��K	GeԞ�A�5*

logging/current_cost6l�;m��+       ��K	��Ԟ�A�5*

logging/current_cost�k�;�:��+       ��K	�Ԟ�A�5*

logging/current_cost�k�;���+       ��K	�Ԟ�A�5*

logging/current_cost�i�;dy.+       ��K	$՞�A�5*

logging/current_cost"n�;���U+       ��K	R՞�A�5*

logging/current_cost
p�;��:+       ��K	(՞�A�5*

logging/current_cost�o�;M�bO+       ��K	z�՞�A�5*

logging/current_cost9m�;��:d+       ��K	j�՞�A�5*

logging/current_costfh�;%GE/+       ��K	�֞�A�5*

logging/current_costAe�;P��+       ��K	�>֞�A�5*

logging/current_costgd�;��0�+       ��K	)n֞�A�5*

logging/current_cost1e�;s�=+       ��K	=�֞�A�5*

logging/current_cost.t�;�/�;+       ��K	��֞�A�5*

logging/current_cost$h�;c6��+       ��K	�֞�A�5*

logging/current_cost�m�;3��Z+       ��K	�'מ�A�6*

logging/current_costWg�;��;I+       ��K	oUמ�A�6*

logging/current_cost.o�;��s+       ��K	��מ�A�6*

logging/current_cost�d�;���+       ��K	�מ�A�6*

logging/current_costa�;h{�+       ��K	�מ�A�6*

logging/current_cost�_�;�t9+       ��K	�؞�A�6*

logging/current_cost�]�;e�^G+       ��K	]=؞�A�6*

logging/current_cost�\�;�+       ��K	sl؞�A�6*

logging/current_cost�`�;�h�+       ��K	��؞�A�6*

logging/current_cost�Z�;rM"�+       ��K	P�؞�A�6*

logging/current_cost�]�;U\+       ��K	�؞�A�6*

logging/current_cost�]�;�	�+       ��K	�)ٞ�A�6*

logging/current_cost�]�;u��+       ��K	�Xٞ�A�6*

logging/current_cost]�;R��+       ��K	�ٞ�A�6*

logging/current_cost�Y�;���+       ��K	صٞ�A�6*

logging/current_cost�\�;�۲+       ��K	��ٞ�A�6*

logging/current_cost^]�;�9{�+       ��K	�ڞ�A�6*

logging/current_costNW�;���|+       ��K	�=ڞ�A�6*

logging/current_costdY�;b^�i+       ��K	�jڞ�A�6*

logging/current_costOV�;1=i+       ��K	�ڞ�A�6*

logging/current_cost�U�;�dz�+       ��K	k�ڞ�A�6*

logging/current_costGW�;S䈔+       ��K	.�ڞ�A�6*

logging/current_cost�T�;{h�+       ��K	~ ۞�A�6*

logging/current_cost�W�; ���+       ��K	JO۞�A�6*

logging/current_cost]V�;C�dw+       ��K	X}۞�A�6*

logging/current_cost�S�;��N+       ��K	�۞�A�7*

logging/current_costPS�;�BA�+       ��K	��۞�A�7*

logging/current_cost�P�;��>V+       ��K	Gܞ�A�7*

logging/current_costfS�;u$,m+       ��K	�1ܞ�A�7*

logging/current_costQ�;/��T+       ��K		`ܞ�A�7*

logging/current_cost O�;;�)+       ��K	ȍܞ�A�7*

logging/current_cost3P�;���+       ��K	�ܞ�A�7*

logging/current_costmR�;����+       ��K	D�ܞ�A�7*

logging/current_cost`T�;���4+       ��K	]$ݞ�A�7*

logging/current_costoN�;ʪ�H+       ��K	�Sݞ�A�7*

logging/current_cost�N�;���+       ��K	��ݞ�A�7*

logging/current_costRJ�;!�ib+       ��K	��ݞ�A�7*

logging/current_cost|J�;���+       ��K	��ݞ�A�7*

logging/current_cost�M�;&��E+       ��K	R
ޞ�A�7*

logging/current_costrY�;mgH+       ��K	G:ޞ�A�7*

logging/current_cost�L�;�+�+       ��K	Gjޞ�A�7*

logging/current_costM�;���+       ��K	t�ޞ�A�7*

logging/current_cost�K�;�&<�+       ��K	Q�ޞ�A�7*

logging/current_cost�F�;v��+       ��K	��ޞ�A�7*

logging/current_costIG�;�D�G+       ��K	�%ߞ�A�7*

logging/current_cost�H�;��~+       ��K	�Uߞ�A�7*

logging/current_cost�J�;�:�+       ��K	�ߞ�A�7*

logging/current_cost*F�;�@��+       ��K	Ʊߞ�A�7*

logging/current_costDH�;�-��+       ��K	(�ߞ�A�7*

logging/current_cost�K�;���y+       ��K	����A�7*

logging/current_cost�I�;�ib=+       ��K	�F���A�7*

logging/current_cost�G�;��ƀ+       ��K	7t���A�8*

logging/current_costSF�;�]�+       ��K	N����A�8*

logging/current_costv@�;>C��+       ��K	_����A�8*

logging/current_costnF�;�6�+       ��K	���A�8*

logging/current_cost�A�;x��+       ��K	�2��A�8*

logging/current_cost \�;�L+       ��K	7d��A�8*

logging/current_cost�C�;V�t+       ��K	���A�8*

logging/current_costm@�; �+       ��K	,���A�8*

logging/current_cost�C�;Ξe�+       ��K	����A�8*

logging/current_cost|@�;�`]+       ��K	��A�8*

logging/current_cost�<�;+f��+       ��K	�L��A�8*

logging/current_cost@=�;�?+       ��K	{��A�8*

logging/current_cost<�;�9�+       ��K	����A�8*

logging/current_cost>�; #�+       ��K	����A�8*

logging/current_cost�=�;���?+       ��K	Z��A�8*

logging/current_cost�;�; �bI+       ��K	`5��A�8*

logging/current_cost	=�;�.�%+       ��K	�c��A�8*

logging/current_cost=:�;'A!�+       ��K	����A�8*

logging/current_costU7�;��S�+       ��K	2���A�8*

logging/current_cost�>�;i���+       ��K	<���A�8*

logging/current_cost�;�;����+       ��K	
��A�8*

logging/current_cost9�;G���+       ��K	5L��A�8*

logging/current_costL:�;rc9k+       ��K	�z��A�8*

logging/current_costC9�;:.+       ��K	����A�8*

logging/current_cost�5�;2:+       ��K	<���A�8*

logging/current_costG8�;ȉ�!+       ��K	���A�8*

logging/current_cost�:�;���+       ��K	�0��A�9*

logging/current_cost9�;���W+       ��K	�_��A�9*

logging/current_cost�5�;�L��+       ��K	����A�9*

logging/current_cost�4�;D�J+       ��K	����A�9*

logging/current_cost1�;Z��1+       ��K	X���A�9*

logging/current_cost3�;��#n+       ��K	
��A�9*

logging/current_cost*2�;��E�+       ��K	�>��A�9*

logging/current_cost�/�;"��P+       ��K	Vm��A�9*

logging/current_costx5�;�8�+       ��K	F���A�9*

logging/current_cost�6�;�W<A+       ��K	����A�9*

logging/current_costw3�;���+       ��K	[���A�9*

logging/current_cost.�;Opt>+       ��K	/(��A�9*

logging/current_cost�,�; d�1+       ��K	�T��A�9*

logging/current_costQ,�;�/��+       ��K	���A�9*

logging/current_cost�0�;�o+       ��K	y���A�9*

logging/current_cost�,�;>���+       ��K	����A�9*

logging/current_cost�,�;�Z�w+       ��K	T��A�9*

logging/current_cost@*�;wN�H+       ��K	Y;��A�9*

logging/current_costM)�;gq��+       ��K	Xj��A�9*

logging/current_cost`,�;��~+       ��K	���A�9*

logging/current_cost�'�;rql+       ��K	����A�9*

logging/current_cost�+�;�F7D+       ��K	����A�9*

logging/current_cost�*�;���+       ��K	���A�9*

logging/current_cost�'�;*���+       ��K	 L��A�9*

logging/current_costC)�;n%Š+       ��K	y��A�9*

logging/current_cost�%�;ݲ�+       ��K	L���A�:*

logging/current_cost'�;�C��+       ��K	����A�:*

logging/current_cost�'�;���,+       ��K	.��A�:*

logging/current_cost�"�;�fR+       ��K	�?��A�:*

logging/current_cost&�;+.�z+       ��K	co��A�:*

logging/current_costp#�;���1+       ��K	s���A�:*

logging/current_cost�!�;pUS/+       ��K	F���A�:*

logging/current_cost�$�;	:��+       ��K	$���A�:*

logging/current_costj#�;Ӑ�+       ��K	H-��A�:*

logging/current_cost�$�;�Ȼi+       ��K	�h��A�:*

logging/current_cost��;��h+       ��K	q���A�:*

logging/current_cost�$�;��*+       ��K	d���A�:*

logging/current_cost	�;"�gy+       ��K	,	��A�:*

logging/current_cost]�;�.�]+       ��K	e9��A�:*

logging/current_cost�!�;@�13+       ��K	el��A�:*

logging/current_costE �;�QJ+       ��K	l���A�:*

logging/current_cost�"�;(m��+       ��K	����A�:*

logging/current_cost��;���"+       ��K	����A�:*

logging/current_costN"�;=+       ��K	M,��A�:*

logging/current_costn�;��3+       ��K	�Z��A�:*

logging/current_costS�;	NZ�+       ��K	.���A�:*

logging/current_cost��;�Զ�+       ��K	ؽ��A�:*

logging/current_cost�;��8�+       ��K	����A�:*

logging/current_cost"�;G�}�+       ��K	<��A�:*

logging/current_cost��;��+       ��K	vK��A�:*

logging/current_cost
�;*&Q+       ��K	�z��A�:*

logging/current_costE�;+��+       ��K	����A�;*

logging/current_cost��;�v:�+       ��K	5���A�;*

logging/current_cost�;�9?�+       ��K	k��A�;*

logging/current_cost�;c%+       ��K	�3��A�;*

logging/current_cost�;���9+       ��K	J`��A�;*

logging/current_cost��;�6�+       ��K	|���A�;*

logging/current_cost��;�{��+       ��K	M���A�;*

logging/current_cost#�;�p}+       ��K	���A�;*

logging/current_costK�;�p�T+       ��K	$��A�;*

logging/current_cost��;���+       ��K	�H��A�;*

logging/current_cost��;�a<+       ��K	hv��A�;*

logging/current_cost��;5��+       ��K	����A�;*

logging/current_cost��;��`+       ��K	=���A�;*

logging/current_costa�;)�08+       ��K	`��A�;*

logging/current_cost��;"6�+       ��K	`/��A�;*

logging/current_cost��;Đ��+       ��K	�]��A�;*

logging/current_costY�;�6W`+       ��K	����A�;*

logging/current_cost�;ٿԛ+       ��K	���A�;*

logging/current_cost/�;-��+       ��K	����A�;*

logging/current_cost6�;�z�+       ��K	��A�;*

logging/current_cost]�;�`7�+       ��K	�A��A�;*

logging/current_cost��;j�`I+       ��K	Mo��A�;*

logging/current_cost	�;#4�a+       ��K	k���A�;*

logging/current_cost��;�l;X+       ��K	q���A�;*

logging/current_cost��;�/�+       ��K	����A�;*

logging/current_costy�;믿�+       ��K	W'��A�<*

logging/current_cost��;!*3�+       ��K	�R��A�<*

logging/current_costE�;�+       ��K	S���A�<*

logging/current_cost��;@�#R+       ��K	����A�<*

logging/current_costH�;$��+       ��K	����A�<*

logging/current_cost��;Wz��+       ��K	�
���A�<*

logging/current_cost�;��A�+       ��K	�9���A�<*

logging/current_cost9�;�y+       ��K	�i���A�<*

logging/current_cost��;�t�+       ��K	�����A�<*

logging/current_cost_�;����+       ��K	�����A�<*

logging/current_costR�;'���+       ��K	J����A�<*

logging/current_cost!�;���b+       ��K	����A�<*

logging/current_cost��;�:�+       ��K	�M���A�<*

logging/current_cost��;j��+       ��K	y���A�<*

logging/current_cost���;�W�+       ��K	5����A�<*

logging/current_cost���;W?�+       ��K	�����A�<*

logging/current_costt �;���+       ��K	z���A�<*

logging/current_cost� �;+@�+       ��K	x1���A�<*

logging/current_cost��;�+       ��K	�^���A�<*

logging/current_cost[ �;.m��+       ��K	}����A�<*

logging/current_cost���;��
+       ��K	N����A�<*

logging/current_cost��;a�+       ��K	|����A�<*

logging/current_cost���;�#hH+       ��K	R(���A�<*

logging/current_cost���;r��{+       ��K	dX���A�<*

logging/current_cost���;7��+       ��K	P����A�<*

logging/current_cost��;����+       ��K	ص���A�<*

logging/current_cost	�;�{}�+       ��K	�����A�=*

logging/current_cost���;�š�+       ��K	����A�=*

logging/current_costQ��;r���+       ��K	�@���A�=*

logging/current_cost���;�״+       ��K	p���A�=*

logging/current_costR��;~�Q+       ��K	�����A�=*

logging/current_cost���;"�+       ��K	�����A�=*

logging/current_cost���;�w1�+       ��K	T����A�=*

logging/current_cost���;�k��+       ��K	�*���A�=*

logging/current_cost���;a��j+       ��K	�W���A�=*

logging/current_cost���;��Ú+       ��K	�����A�=*

logging/current_cost��;7t+       ��K	2����A�=*

logging/current_cost���;�`�.+       ��K	�����A�=*

logging/current_cost\��;���Z+       ��K	����A�=*

logging/current_costX�;Ļ+       ��K	=I���A�=*

logging/current_cost��;�%�x+       ��K	4z���A�=*

logging/current_cost��;3w�+       ��K	�����A�=*

logging/current_cost��;IK+       ��K	�����A�=*

logging/current_costZ�;h���+       ��K		���A�=*

logging/current_cost �;�`�+       ��K	�7���A�=*

logging/current_cost��;��+       ��K	�e���A�=*

logging/current_cost���;�!#�+       ��K	ӭ���A�=*

logging/current_cost4�;6E�+       ��K	x����A�=*

logging/current_cost6�;0��+       ��K	7	���A�=*

logging/current_cost��;��+       ��K	�5���A�=*

logging/current_cost�;*�i+       ��K	�e���A�=*

logging/current_cost
�;�I�E+       ��K	ۖ���A�=*

logging/current_cost��;���+       ��K	�����A�>*

logging/current_costD�;�d��+       ��K	-����A�>*

logging/current_cost��;1��+       ��K	� ���A�>*

logging/current_cost��;
�d`+       ��K	O���A�>*

logging/current_cost_�;�7��+       ��K	~���A�>*

logging/current_cost��;.䅊+       ��K	m����A�>*

logging/current_cost��;��ش+       ��K	.����A�>*

logging/current_cost��;,s�W+       ��K	@
���A�>*

logging/current_cost���;�`�~+       ��K	�7���A�>*

logging/current_cost��;�*��+       ��K	�c���A�>*

logging/current_cost�;��O{+       ��K	�����A�>*

logging/current_cost�;/c�+       ��K	�����A�>*

logging/current_cost�;�'�+       ��K	�����A�>*

logging/current_cost��;+<Z�+       ��K	v���A�>*

logging/current_cost��;�1-s+       ��K	�M���A�>*

logging/current_costs�;�Pa�+       ��K	�~���A�>*

logging/current_cost��;�*�q+       ��K	�����A�>*

logging/current_cost�;+⷇+       ��K	�����A�>*

logging/current_cost_�;����+       ��K	V ��A�>*

logging/current_cost���;���L+       ��K	~4 ��A�>*

logging/current_costv�;ڮ
A+       ��K	7c ��A�>*

logging/current_costW�;R|�X+       ��K	�� ��A�>*

logging/current_coste�;���+       ��K	s� ��A�>*

logging/current_cost��;�!�b+       ��K	)� ��A�>*

logging/current_cost��;>}V�+       ��K	B��A�>*

logging/current_cost��;k�U�+       ��K	�J��A�?*

logging/current_cost���;�$<�+       ��K	y��A�?*

logging/current_cost��;�Eї+       ��K	v���A�?*

logging/current_costk߁;��t+       ��K	����A�?*

logging/current_cost��;�FN+       ��K	���A�?*

logging/current_cost#ށ;V�R+       ��K	w0��A�?*

logging/current_cost�݁;>�mi+       ��K	]��A�?*

logging/current_cost!݁;��4+       ��K	E���A�?*

logging/current_cost܁;l~-H+       ��K	���A�?*

logging/current_cost�ہ;�,Z�+       ��K	����A�?*

logging/current_cost���;��&+       ��K	D��A�?*

logging/current_cost�ف;�~(�+       ��K	{@��A�?*

logging/current_cost�݁;��6:+       ��K	5q��A�?*

logging/current_cost�ہ;U��+       ��K	Ԧ��A�?*

logging/current_cost5ځ;��S#+       ��K	E���A�?*

logging/current_costQׁ;�qA�+       ��K	W��A�?*

logging/current_cost�ف;wަ1+       ��K	�H��A�?*

logging/current_costaف;IǬ,+       ��K	}{��A�?*

logging/current_cost{ց;�"F�+       ��K	���A�?*

logging/current_costiׁ;�q}F+       ��K	`���A�?*

logging/current_cost�ց;Q��+       ��K	�#��A�?*

logging/current_costbׁ;��v+       ��K	)Z��A�?*

logging/current_cost�с;�L+       ��K	n���A�?*

logging/current_cost܁;��{"+       ��K	���A�?*

logging/current_cost�ہ;	�+       ��K	���A�?*

logging/current_costeс;e�4�+       ��K	�/��A�?*

logging/current_cost!Ӂ;p�,�+       ��K	�b��A�@*

logging/current_costс;��K�+       ��K	���A�@*

logging/current_cost%Ձ;l	�+       ��K	���A�@*

logging/current_cost*ځ;ht�+       ��K	� ��A�@*

logging/current_costtԁ;���+       ��K	�7��A�@*

logging/current_costՁ;�Ƃ�+       ��K	Cl��A�@*

logging/current_cost�с;��+       ��K	<���A�@*

logging/current_cost;ف;�J�+       ��K	t���A�@*

logging/current_cost�́;�I�s+       ��K	���A�@*

logging/current_cost�Ӂ; (j�+       ��K	�:��A�@*

logging/current_costbс;){#�+       ��K	�n��A�@*

logging/current_cost�ˁ;o�z+       ��K	����A�@*

logging/current_cost�ʁ;�W�w+       ��K	����A�@*

logging/current_cost�ʁ;nZ�a+       ��K	1 	��A�@*

logging/current_cost�ȁ;��SZ+       ��K	�0	��A�@*

logging/current_cost�Ձ;�=	~+       ��K	 g	��A�@*

logging/current_cost�́;��-+       ��K	��	��A�@*

logging/current_costsˁ;�Z�|+       ��K	C�	��A�@*

logging/current_cost�ρ;T��+       ��K	�
��A�@*

logging/current_cost�ȁ;L�Q�+       ��K	�9
��A�@*

logging/current_cost�ˁ;=EX6+       ��K	�o
��A�@*

logging/current_costHȁ;MG�#+       ��K	��
��A�@*

logging/current_costjρ;��]2+       ��K	_�
��A�@*

logging/current_costGā;��hi+       ��K	���A�@*

logging/current_cost7́;r���+       ��K	�B��A�@*

logging/current_cost1ȁ;�p�+       ��K	�q��A�A*

logging/current_costŁ;?�j�+       ��K	����A�A*

logging/current_cost́;�w�+       ��K	����A�A*

logging/current_cost���;��d+       ��K	����A�A*

logging/current_cost%Ł;P�?_+       ��K	)��A�A*

logging/current_cost�ā;s�~+       ��K	�X��A�A*

logging/current_cost�Á;��8�+       ��K	Ć��A�A*

logging/current_costAƁ;���+       ��K	A���A�A*

logging/current_cost�Ɓ;T��+       ��K	���A�A*

logging/current_cost���;��t�+       ��K	���A�A*

logging/current_cost��;+�E�+       ��K	7>��A�A*

logging/current_cost�ȁ;n�p�+       ��K	�p��A�A*

logging/current_cost8��;�?V�+       ��K	���A�A*

logging/current_cost���;����+       ��K	���A�A*

logging/current_costw��;`�+       ��K	����A�A*

logging/current_cost1��;���I+       ��K	R(��A�A*

logging/current_cost���;m��+       ��K	W��A�A*

logging/current_cost�;�H+       ��K	_���A�A*

logging/current_cost���;O���+       ��K	#���A�A*

logging/current_costh��;��#+       ��K	����A�A*

logging/current_cost�ā;m���+       ��K	���A�A*

logging/current_cost^с;6���+       ��K	�=��A�A*

logging/current_cost��;���R+       ��K	�i��A�A*

logging/current_cost0��;���+       ��K	w���A�A*

logging/current_cost<��;Γa.+       ��K	����A�A*

logging/current_cost��;�y
[+       ��K	?1��A�A*

logging/current_cost7��;��5+       ��K	ju��A�B*

logging/current_cost���;ᔂ�+       ��K	G���A�B*

logging/current_costb��;�}U�+       ��K	����A�B*

logging/current_cost?��;�'+       ��K	�"��A�B*

logging/current_cost���;��e+       ��K	�[��A�B*

logging/current_costӹ�;6!͕+       ��K	V���A�B*

logging/current_costH��;)Iz�+       ��K	����A�B*

logging/current_costN��;�L��+       ��K	}��A�B*

logging/current_cost��;��++       ��K	�s��A�B*

logging/current_cost��;�=�+       ��K	����A�B*

logging/current_cost1��;�A�+       ��K	"���A�B*

logging/current_cost���;]�G�+       ��K	DJ��A�B*

logging/current_costѲ�;���+       ��K	���A�B*

logging/current_cost���;�:�^+       ��K	h���A�B*

logging/current_cost���;{E�?+       ��K	����A�B*

logging/current_costു;(�+       ��K	a4��A�B*

logging/current_cost���;�mm�+       ��K	Uj��A�B*

logging/current_cost{��;��З+       ��K	����A�B*

logging/current_costn��;���+       ��K	���A�B*

logging/current_costT��;L��s+       ��K	����A�B*

logging/current_costǬ�;�N�_+       ��K	7��A�B*

logging/current_cost�;�o��+       ��K	�z��A�B*

logging/current_costf��;2�%,+       ��K	ϫ��A�B*

logging/current_cost3��;�wش+       ��K	���A�B*

logging/current_cost"��;�Ҹ+       ��K	o��A�B*

logging/current_cost��;'MiM+       ��K	�Y��A�B*

logging/current_costT��;���+       ��K	����A�C*

logging/current_costѯ�;��+       ��K	����A�C*

logging/current_cost���;�_1�+       ��K	����A�C*

logging/current_cost3��;1 4�+       ��K	/.��A�C*

logging/current_cost��;�`�X+       ��K	.c��A�C*

logging/current_costʬ�;?JI�+       ��K	l���A�C*

logging/current_cost���;�	G/+       ��K	I���A�C*

logging/current_costǲ�;��Di+       ��K	H���A�C*

logging/current_costE��;�V@�+       ��K	�4��A�C*

logging/current_costv��;�^��+       ��K	h��A�C*

logging/current_cost���;��+       ��K	���A�C*

logging/current_cost���;?�'+       ��K	)���A�C*

logging/current_cost~��;\G��+       ��K	C���A�C*

logging/current_cost���;C�?�+       ��K	�4��A�C*

logging/current_costܧ�;�s�+       ��K	�e��A�C*

logging/current_costw��;.�(~+       ��K	���A�C*

logging/current_costȥ�;�0Ӗ+       ��K	����A�C*

logging/current_costN��;��5
+       ��K	���A�C*

logging/current_costc��;x+       ��K	78��A�C*

logging/current_cost�;�"�e+       ��K	`i��A�C*

logging/current_costΦ�;�Ở+       ��K	u���A�C*

logging/current_cost�;�\G+       ��K	����A�C*

logging/current_cost;��;��j�+       ��K	����A�C*

logging/current_costf��;��]�+       ��K	�-��A�C*

logging/current_cost쟁;�PG4+       ��K	�a��A�C*

logging/current_cost���;w*Hi+       ��K	X���A�D*

logging/current_cost�ށ;���+       ��K	n���A�D*

logging/current_cost���;����+       ��K	B���A�D*

logging/current_costb��;vV(�+       ��K	K&��A�D*

logging/current_costī�;�[7�+       ��K	�U��A�D*

logging/current_cost@��;!u�t+       ��K	T���A�D*

logging/current_costo��;K�:+       ��K		���A�D*

logging/current_costC��;����+       ��K	���A�D*

logging/current_costx��;��+]+       ��K	?��A�D*

logging/current_cost)��;�/H%+       ��K	�y��A�D*

logging/current_cost���;���+       ��K	L���A�D*

logging/current_costC��;|�=+       ��K	j���A�D*

logging/current_costy��;��Ư+       ��K	���A�D*

logging/current_cost��;P���+       ��K	mR��A�D*

logging/current_cost���;�lOu+       ��K	���A�D*

logging/current_costD��;��,b+       ��K	����A�D*

logging/current_cost��;�n�3+       ��K	����A�D*

logging/current_cost8��;C~��+       ��K	�2��A�D*

logging/current_cost���;�S��+       ��K	�d��A�D*

logging/current_costY��;���+       ��K	����A�D*

logging/current_cost���;��jW+       ��K	0���A�D*

logging/current_cost'��;��+       ��K	w ��A�D*

logging/current_cost���;�Yѡ+       ��K	�@ ��A�D*

logging/current_cost!��;�c�+       ��K	�w ��A�D*

logging/current_cost���;�n��+       ��K	�� ��A�D*

logging/current_cost���;"�P�+       ��K	�� ��A�D*

logging/current_cost��;0 b�+       ��K	!��A�E*

logging/current_cost㔁;(Y�+       ��K	�>!��A�E*

logging/current_costG��;8?ܺ+       ��K	�p!��A�E*

logging/current_costC��;��+       ��K	"�!��A�E*

logging/current_cost���;,�Ը+       ��K	��!��A�E*

logging/current_cost)��;�|��+       ��K	�"��A�E*

logging/current_cost��;��+       ��K	�C"��A�E*

logging/current_cost��;�+       ��K	Tv"��A�E*

logging/current_cost���;�A8�+       ��K	��"��A�E*

logging/current_cost���;"p��+       ��K	��"��A�E*

logging/current_cost���;�Bu+       ��K	�#��A�E*

logging/current_cost���;]��+       ��K	�J#��A�E*

logging/current_costL��;�� �+       ��K	�z#��A�E*

logging/current_costԏ�;Eċ+       ��K	��#��A�E*

logging/current_cost���;��l�+       ��K	��#��A�E*

logging/current_cost/��;-b�R+       ��K	$��A�E*

logging/current_cost���;��k�+       ��K		J$��A�E*

logging/current_cost��;����+       ��K	z$��A�E*

logging/current_costb��;J<V�+       ��K	�$��A�E*

logging/current_cost7��;�a�+       ��K	��$��A�E*

logging/current_cost���;P�,�+       ��K	%��A�E*

logging/current_cost���;̹�+       ��K	cY%��A�E*

logging/current_costa��;+Я�+       ��K	��%��A�E*

logging/current_costӉ�;#�ܪ+       ��K	�%��A�E*

logging/current_cost9��;�l?+       ��K	 &��A�E*

logging/current_cost}��;9�]+       ��K	
0&��A�F*

logging/current_costӜ�;��gz+       ��K	?a&��A�F*

logging/current_cost��;z)"�+       ��K	�&��A�F*

logging/current_cost���;y�+       ��K	B�&��A�F*

logging/current_cost���;_L+       ��K	��&��A�F*

logging/current_costT��;C�	+       ��K	g%'��A�F*

logging/current_cost7��;�x�.+       ��K	U'��A�F*

logging/current_cost?��;�ұl+       ��K	��'��A�F*

logging/current_costt��;���+       ��K	x�'��A�F*

logging/current_costM��;L� +       ��K	��'��A�F*

logging/current_cost˄�;��|_+       ��K	u(��A�F*

logging/current_costW��;�^za+       ��K	�>(��A�F*

logging/current_cost���;�R�\+       ��K	�m(��A�F*

logging/current_costo��;�ۀ�+       ��K	�(��A�F*

logging/current_costn��;��+       ��K	��(��A�F*

logging/current_costE��;sEm +       ��K	��(��A�F*

logging/current_costD��;N�g+       ��K	�))��A�F*

logging/current_cost���;�JX�+       ��K	�X)��A�F*

logging/current_cost��;��҉+       ��K	��)��A�F*

logging/current_cost��;�P�+       ��K	��)��A�F*

logging/current_cost9��;%��h+       ��K	\�)��A�F*

logging/current_costP��;
��+       ��K	�*��A�F*

logging/current_cost^��;Y'x�+       ��K	�8*��A�F*

logging/current_costB��;:]�k+       ��K	%f*��A�F*

logging/current_cost��;g ŕ+       ��K	��*��A�F*

logging/current_cost�|�;.ί�+       ��K	��*��A�F*

logging/current_cost҄�;�n+       ��K	r�*��A�G*

logging/current_costT��;|X88+       ��K	l'+��A�G*

logging/current_costA��;�2'9+       ��K	PZ+��A�G*

logging/current_cost7~�;��a�+       ��K	Ë+��A�G*

logging/current_cost;��;�w�+       ��K	�+��A�G*

logging/current_cost^��;�}� +       ��K	��+��A�G*

logging/current_cost���;y��+       ��K	�%,��A�G*

logging/current_costh{�;&PU]+       ��K	V,��A�G*

logging/current_cost�~�;��M�+       ��K	��,��A�G*

logging/current_costb�;ݘ�8+       ��K	V�,��A�G*

logging/current_cost��;SD��+       ��K	_�,��A�G*

logging/current_cost�~�;{@++       ��K	�!-��A�G*

logging/current_cost�|�;��H0+       ��K	S-��A�G*

logging/current_cost\~�;��+       ��K	A�-��A�G*

logging/current_costPx�;݊n�+       ��K	��-��A�G*

logging/current_cost�z�;�9�+       ��K	�-��A�G*

logging/current_cost���;^��+       ��K	h.��A�G*

logging/current_cost|�;��f�+       ��K	s?.��A�G*

logging/current_cost	|�;��R{+       ��K	�l.��A�G*

logging/current_cost�t�;҆ݮ+       ��K	̟.��A�G*

logging/current_cost�s�;Әaf+       ��K	��.��A�G*

logging/current_cost�y�;Q���+       ��K	+�.��A�G*

logging/current_cost-w�;�C4�+       ��K	M-/��A�G*

logging/current_cost}�;��b+       ��K	�[/��A�G*

logging/current_costr��;uYh+       ��K	�/��A�G*

logging/current_cost~�;�&N�+       ��K	��/��A�G*

logging/current_costu�;5�V�+       ��K	$�/��A�H*

logging/current_cost~��;|�%%+       ��K	]%0��A�H*

logging/current_costz�;�T�+       ��K	�T0��A�H*

logging/current_cost�p�;y:�8+       ��K	��0��A�H*

logging/current_costSw�;T�*f+       ��K	l�0��A�H*

logging/current_cost�v�; 1+       ��K	=�0��A�H*

logging/current_costDp�;yv+       ��K	�1��A�H*

logging/current_cost�q�;�v�M+       ��K	(H1��A�H*

logging/current_cost�n�;��F+       ��K	�v1��A�H*

logging/current_cost�q�;:+       ��K	�1��A�H*

logging/current_cost�n�;����+       ��K	M�1��A�H*

logging/current_costbr�;�r�+       ��K	�2��A�H*

logging/current_cost�n�;�oi�+       ��K	k02��A�H*

logging/current_cost�q�;xe�+       ��K	�]2��A�H*

logging/current_cost�r�;��Nw+       ��K	��2��A�H*

logging/current_costk�;9���+       ��K	�2��A�H*

logging/current_cost�o�;��E+       ��K	��2��A�H*

logging/current_cost�q�;Vŝ�+       ��K	�3��A�H*

logging/current_cost�p�;o��+       ��K	%G3��A�H*

logging/current_costl�;�A��+       ��K	�s3��A�H*

logging/current_cost�j�;��4+       ��K	.�3��A�H*

logging/current_costHq�;# � +       ��K	5�3��A�H*

logging/current_costUj�;���+       ��K	��3��A�H*

logging/current_costkk�;��+       ��K	#(4��A�H*

logging/current_cost�l�;d�V++       ��K	]V4��A�H*

logging/current_cost�j�;HYͅ+       ��K	<�4��A�I*

logging/current_cost�g�;*���+       ��K	d�4��A�I*

logging/current_costxf�;w\�+       ��K	��4��A�I*

logging/current_costgh�;\���+       ��K	B5��A�I*

logging/current_cost�f�;��0+       ��K	uA5��A�I*

logging/current_cost1h�;w�,+       ��K	�p5��A�I*

logging/current_cost�d�;�D��+       ��K	?�5��A�I*

logging/current_costh�;�3+       ��K	��5��A�I*

logging/current_cost�e�;� [�+       ��K	�5��A�I*

logging/current_cost�f�;Q���+       ��K	�)6��A�I*

logging/current_cost�f�;���+       ��K	�X6��A�I*

logging/current_cost<e�;�ħ+       ��K	I�6��A�I*

logging/current_cost�d�;���C+       ��K	��6��A�I*

logging/current_cost9c�;&�+       ��K	�6��A�I*

logging/current_cost�p�;Ԇ�+       ��K	i7��A�I*

logging/current_cost�v�;>��F+       ��K	\?7��A�I*

logging/current_cost�e�;�,+       ��K	�n7��A�I*

logging/current_cost�e�;Fy�@+       ��K	:�7��A�I*

logging/current_costig�;zH��+       ��K	
�7��A�I*

logging/current_costq`�;2���+       ��K	G8��A�I*

logging/current_cost�b�;)��+       ��K	�;8��A�I*

logging/current_cost#`�;� Û+       ��K	+k8��A�I*

logging/current_costK`�;�X4�+       ��K	l�8��A�I*

logging/current_cost7_�;`��j+       ��K	�8��A�I*

logging/current_cost�i�;��,�+       ��K	��8��A�I*

logging/current_cost[`�;�Wi+       ��K	�&9��A�I*

logging/current_costha�;��� +       ��K	3X9��A�J*

logging/current_cost�^�;�7+       ��K	��9��A�J*

logging/current_cost�f�;5���+       ��K	��9��A�J*

logging/current_cost-`�;E��+       ��K	��9��A�J*

logging/current_coste\�;F�Y�+       ��K	*:��A�J*

logging/current_costz_�;%�yL+       ��K	�F:��A�J*

logging/current_cost\�;E�^�+       ��K	u:��A�J*

logging/current_costoj�;�^\+       ��K	�:��A�J*

logging/current_cost Z�;�\�+       ��K	��:��A�J*

logging/current_cost�g�;	q�6+       ��K	i;��A�J*

logging/current_cost	_�;��+       ��K	�4;��A�J*

logging/current_costI]�;S�]+       ��K	{g;��A�J*

logging/current_cost X�;��]	+       ��K	M�;��A�J*

logging/current_costrZ�;a�~W+       ��K	7<��A�J*

logging/current_cost�X�;a-:?+       ��K	��<��A�J*

logging/current_cost�W�;�Ӎ+       ��K	a�<��A�J*

logging/current_cost�Y�;V#��+       ��K	�=��A�J*

logging/current_costMY�;�yo +       ��K	�==��A�J*

logging/current_cost�]�;���+       ��K	"r=��A�J*

logging/current_cost�[�;�u1+       ��K	�=��A�J*

logging/current_cost�Y�;B��+       ��K	k�=��A�J*

logging/current_costSX�;���"+       ��K	f>��A�J*

logging/current_costbY�;��~A+       ��K	�=>��A�J*

logging/current_cost�V�;^O+       ��K	6n>��A�J*

logging/current_costvV�;GӌS+       ��K	�>��A�J*

logging/current_cost[�;�m��+       ��K	}�>��A�K*

logging/current_cost�V�;�t+       ��K	�?��A�K*

logging/current_costU�;t]�5+       ��K	m4?��A�K*

logging/current_cost4U�;4W�+       ��K	fe?��A�K*

logging/current_cost X�;���A+       ��K	ݙ?��A�K*

logging/current_cost!X�;�U�g+       ��K	(�?��A�K*

logging/current_costS�;h5�++       ��K	��?��A�K*

logging/current_cost�Q�;7j��+       ��K	�(@��A�K*

logging/current_cost�V�;��+�+       ��K	�V@��A�K*

logging/current_costDS�;Bydp+       ��K	v�@��A�K*

logging/current_cost�R�;�E�+       ��K	��@��A�K*

logging/current_cost�R�;cK^�+       ��K	�@��A�K*

logging/current_cost�Q�;�M#+       ��K	�A��A�K*

logging/current_cost}T�;�s�+       ��K	W<A��A�K*

logging/current_cost5O�;��r\+       ��K	VjA��A�K*

logging/current_cost�N�;��It+       ��K	X�A��A�K*

logging/current_cost�T�;BsdF+       ��K	�A��A�K*

logging/current_cost�Q�;Au8+       ��K	��A��A�K*

logging/current_costP�;�V��+       ��K	h0B��A�K*

logging/current_cost�O�;���W+       ��K	��B��A�K*

logging/current_cost�O�;�=�l+       ��K	\�B��A�K*

logging/current_cost�M�;4NJ+       ��K	��B��A�K*

logging/current_costZP�;K-�+       ��K	�3C��A�K*

logging/current_costM�;6��+       ��K	�jC��A�K*

logging/current_cost)K�;���+       ��K	��C��A�K*

logging/current_costbP�;��%+       ��K	)�C��A�K*

logging/current_cost�P�; ݼ�+       ��K	�D��A�L*

logging/current_cost)W�;�,�+       ��K	�BD��A�L*

logging/current_costkM�;�B> +       ��K	�qD��A�L*

logging/current_cost'J�;�G�+       ��K	'�D��A�L*

logging/current_cost�M�;<�V+       ��K	��D��A�L*

logging/current_costK�;�]5�+       ��K	�E��A�L*

logging/current_costK�;�"��+       ��K	�4E��A�L*

logging/current_cost	G�;G�+       ��K	)eE��A�L*

logging/current_cost�]�;�+��+       ��K	�E��A�L*

logging/current_coste�;
�C+       ��K	��E��A�L*

logging/current_cost�T�;$��+       ��K	��E��A�L*

logging/current_costWN�;�żQ+       ��K	!F��A�L*

logging/current_cost�K�;P�C�+       ��K	�SF��A�L*

logging/current_cost�I�;�I�+       ��K	g�F��A�L*

logging/current_costXD�;/.��+       ��K	�F��A�L*

logging/current_cost�G�;�\��+       ��K	�F��A�L*

logging/current_cost3D�;��Nj+       ��K	�G��A�L*

logging/current_costuE�;`D?+       ��K	�<G��A�L*

logging/current_cost�E�;�S�}+       ��K	�lG��A�L*

logging/current_costtE�;��[M+       ��K	șG��A�L*

logging/current_costsD�;]�)\+       ��K	��G��A�L*

logging/current_cost�A�;cס+       ��K	F�G��A�L*

logging/current_cost�D�;���g+       ��K	9H��A�L*

logging/current_costGA�;�_��+       ��K	$rH��A�L*

logging/current_cost}C�;@t;+       ��K	��H��A�L*

logging/current_cost�?�;�+       ��K	��H��A�L*

logging/current_costNE�;fY�8+       ��K	EI��A�M*

logging/current_costCA�;�\�+       ��K	z9I��A�M*

logging/current_cost�E�;)��+       ��K	��I��A�M*

logging/current_cost�F�;dFa+       ��K	��I��A�M*

logging/current_cost�E�;~��!+       ��K	�J��A�M*

logging/current_cost�@�;��*+       ��K	�NJ��A�M*

logging/current_cost�@�;�LL�+       ��K	��J��A�M*

logging/current_cost�F�;�H"+       ��K	��J��A�M*

logging/current_cost_G�;t��+       ��K	��J��A�M*

logging/current_cost�A�;'��-+       ��K	9>K��A�M*

logging/current_cost'?�;Ca�X+       ��K	q|K��A�M*

logging/current_cost>�;�#�+       ��K	��K��A�M*

logging/current_cost�C�;��+       ��K	�K��A�M*

logging/current_costB�;|���+       ��K	�L��A�M*

logging/current_costE�;zP�+       ��K	�QL��A�M*

logging/current_costQ<�;�C�y+       ��K	�L��A�M*

logging/current_cost�>�;n�^�+       ��K	��L��A�M*

logging/current_cost�:�;�0o +       ��K	%�L��A�M*

logging/current_costP:�;���S+       ��K	�0M��A�M*

logging/current_cost�9�;<��g+       ��K	PjM��A�M*

logging/current_costZ9�;���+       ��K	}�M��A�M*

logging/current_cost9�;��W�+       ��K	��M��A�M*

logging/current_costN9�;�\A+       ��K	�N��A�M*

logging/current_cost�8�;�d�~+       ��K	&CN��A�M*

logging/current_costc9�;��i>+       ��K	�pN��A�M*

logging/current_cost�7�;+�+       ��K	N��A�N*

logging/current_cost�F�;60�+       ��K	��N��A�N*

logging/current_cost)E�;:�#�+       ��K	��N��A�N*

logging/current_cost	6�;����