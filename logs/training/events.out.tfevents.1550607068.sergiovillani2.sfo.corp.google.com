       �K"	   ��Abrain.Event:2l��M�      ��	XQ6��A"��
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
9layer_1/weights1/Initializer/random_uniform/RandomUniformRandomUniform1layer_1/weights1/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_1/weights1*
seed2 
�
/layer_1/weights1/Initializer/random_uniform/subSub/layer_1/weights1/Initializer/random_uniform/max/layer_1/weights1/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@layer_1/weights1
�
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
layer_1/weights1
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
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
layer_1/weights1/readIdentitylayer_1/weights1*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
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
layer_1/biases1/readIdentitylayer_1/biases1*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
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
layer_1/ReluRelulayer_1/add*'
_output_shapes
:���������*
T0
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

seed *
T0*#
_class
loc:@layer_2/weights2*
seed2 *
dtype0*
_output_shapes

:
�
/layer_2/weights2/Initializer/random_uniform/subSub/layer_2/weights2/Initializer/random_uniform/max/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
layer_2/weights2
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
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2
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
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j
layer_2/addAddlayer_2/MatMullayer_2/biases2/read*
T0*'
_output_shapes
:���������
S
layer_2/ReluRelulayer_2/add*'
_output_shapes
:���������*
T0
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
!layer_3/biases3/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
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
layer_3/ReluRelulayer_3/add*'
_output_shapes
:���������*
T0
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
.output/weights4/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@output/weights4*
valueB
 *qĜ?
�
8output/weights4/Initializer/random_uniform/RandomUniformRandomUniform0output/weights4/Initializer/random_uniform/shape*
T0*"
_class
loc:@output/weights4*
seed2 *
dtype0*
_output_shapes

:*

seed 
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
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
output/weights4
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
output/weights4/AssignAssignoutput/weights4*output/weights4/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
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
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container 
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
cost/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
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
$train/gradients/cost/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
'train/gradients/output/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
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
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
0train/gradients/output/add_grad/tuple/group_depsNoOp(^train/gradients/output/add_grad/Reshape*^train/gradients/output/add_grad/Reshape_1
�
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape
�
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1*
_output_shapes
:
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
�
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul
�
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1
�
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*'
_output_shapes
:���������*
T0
t
&train/gradients/layer_3/add_grad/ShapeShapelayer_3/MatMul*
_output_shapes
:*
T0*
out_type0
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
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul
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
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
T0*
out_type0*
_output_shapes
:
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
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
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
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape
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
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
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
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1
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
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
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
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1
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
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1
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
VariableV2*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

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
,train/layer_1/biases1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_1/biases1*
valueB*    
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
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam_1
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
-train/layer_2/weights2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_2/weights2*
valueB*    
�
train/layer_2/weights2/Adam
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
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
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
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2
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
"train/layer_3/weights3/Adam_1/readIdentitytrain/layer_3/weights3/Adam_1*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
,train/layer_3/biases3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_3/biases3*
valueB*    
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
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
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
,train/output/weights4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*"
_class
loc:@output/weights4*
valueB*    
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
train/output/weights4/Adam/readIdentitytrain/output/weights4/Adam*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
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
-train/output/biases4/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
�
train/output/biases4/Adam_1
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
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
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
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@layer_1/weights1*
use_nesterov( *
_output_shapes

:*
use_locking( 
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
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_3/biases3*
use_nesterov( *
_output_shapes
:
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
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking( 
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
save/SaveV2/tensor_namesConst*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
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
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*(
dtypes
2*|
_output_shapesj
h::::::::::::::::::::::::::
�
save/AssignAssignlayer_1/biases1save/RestoreV2*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
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
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
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
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
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
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
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
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
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
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
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
save/Assign_22Assigntrain/output/biases4/Adamsave/RestoreV2:22*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
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
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign"/ײ��     ��d]	+<;��AJ܉
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
dtype0*'
_output_shapes
:���������*
shape:���������
�
1layer_1/weights1/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@layer_1/weights1*
valueB"      
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
/layer_1/weights1/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_1/weights1*
valueB
 *�7�?
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
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
layer_1/weights1
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
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
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
layer_1/biases1/readIdentitylayer_1/biases1*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
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

seed *
T0*#
_class
loc:@layer_2/weights2*
seed2 *
dtype0*
_output_shapes

:
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
VariableV2*"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
z
layer_2/biases2/readIdentitylayer_2/biases2*
_output_shapes
:*
T0*"
_class
loc:@layer_2/biases2
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
1layer_3/weights3/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_3/weights3*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_3/weights3/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_3/weights3*
valueB
 *׳]�
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
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@layer_3/weights3
�
/layer_3/weights3/Initializer/random_uniform/mulMul9layer_3/weights3/Initializer/random_uniform/RandomUniform/layer_3/weights3/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
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
layer_3/weights3/readIdentitylayer_3/weights3*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
!layer_3/biases3/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
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
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
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
0output/weights4/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@output/weights4*
valueB"      
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
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
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
dtype0*'
_output_shapes
:���������*
shape:���������
{
cost/SquaredDifferenceSquaredDifference
output/addcost/Placeholder*'
_output_shapes
:���������*
T0
[

cost/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
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
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
|
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
_output_shapes
:*
T0*
out_type0
i
&train/gradients/cost/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/cost/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
p
&train/gradients/cost/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
'train/gradients/cost/Mean_grad/floordivFloorDiv#train/gradients/cost/Mean_grad/Prod&train/gradients/cost/Mean_grad/Maximum*
_output_shapes
: *
T0
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
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
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
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
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
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul
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
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
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
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
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
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*'
_output_shapes
:���������*
T0
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
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
�
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
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
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
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
-train/layer_1/weights1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_1/weights1*
valueB*    
�
train/layer_1/weights1/Adam
VariableV2*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
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
,train/layer_1/biases1/Adam/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam
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
VariableV2*"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
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
!train/layer_1/biases1/Adam_1/readIdentitytrain/layer_1/biases1/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
�
-train/layer_2/weights2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_2/weights2*
valueB*    
�
train/layer_2/weights2/Adam
VariableV2*#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
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
 train/layer_2/weights2/Adam/readIdentitytrain/layer_2/weights2/Adam*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
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
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@layer_2/biases2
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
VariableV2*#
_class
loc:@layer_3/weights3*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
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
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
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
train/layer_3/biases3/Adam/readIdentitytrain/layer_3/biases3/Adam*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam_1
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
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
�
,train/output/weights4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*"
_class
loc:@output/weights4*
valueB*    
�
train/output/weights4/Adam
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
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
!train/output/weights4/Adam_1/readIdentitytrain/output/weights4/Adam_1*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
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
-train/output/biases4/Adam_1/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
train/output/biases4/Adam_1
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
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
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
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@layer_1/weights1*
use_nesterov( *
_output_shapes

:*
use_locking( 
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
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@layer_2/weights2*
use_nesterov( *
_output_shapes

:*
use_locking( 
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
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_3/biases3*
use_nesterov( *
_output_shapes
:
�
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*"
_class
loc:@output/weights4
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
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
_output_shapes
: *
T0*"
_class
loc:@layer_1/biases1
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
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
_output_shapes
: *
T0*"
_class
loc:@layer_1/biases1
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

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
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
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
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
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_2Assignlayer_2/biases2save/RestoreV2:2*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
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
save/Assign_7Assignoutput/weights4save/RestoreV2:7*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
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
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
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
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
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
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
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
save/Assign_22Assigntrain/output/biases4/Adamsave/RestoreV2:22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
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
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/Assign_25Assigntrain/output/weights4/Adam_1save/RestoreV2:25*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign""
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
train/output/biases4/Adam_1:0"train/output/biases4/Adam_1/Assign"train/output/biases4/Adam_1/read:02/train/output/biases4/Adam_1/Initializer/zeros:0"�
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
logging/current_cost:0�-��(       �pJ	ga@��A*

logging/current_cost��=d<�*       ����	�@��A*

logging/current_cost���=+5)f*       ����	P�@��A
*

logging/current_cost��=�4�w*       ����	��@��A*

logging/current_costO�=;|qS*       ����	�+A��A*

logging/current_cost9{=wC�f*       ����	o[A��A*

logging/current_costK�Y=�TѬ*       ����	u�A��A*

logging/current_cost3@=�k��*       ����	$�A��A#*

logging/current_cost�,=6�*       ����	f�A��A(*

logging/current_cost�\=���*       ����	*B��A-*

logging/current_cost�=���*       ����	�`B��A2*

logging/current_cost��=�K �*       ����	1�B��A7*

logging/current_cost�o�<�5].*       ����	��B��A<*

logging/current_cost׌�<Mw��*       ����	;�B��AA*

logging/current_cost�0�<�v*       ����	�#C��AF*

logging/current_cost�O�<�J��*       ����	�SC��AK*

logging/current_costI��<�>ls*       ����	G�C��AP*

logging/current_costg�<$t*J*       ����	��C��AU*

logging/current_cost)�<��'*       ����	r�C��AZ*

logging/current_costI�<�:�*       ����	�D��A_*

logging/current_costF��<���)*       ����	c:D��Ad*

logging/current_cost��<����*       ����	�iD��Ai*

logging/current_cost�@v<dZ�!*       ����	��D��An*

logging/current_cost\�g<�@H*       ����	4�D��As*

logging/current_cost��Y<�/�*       ����	{�D��Ax*

logging/current_cost<\M<N���*       ����	�#E��A}*

logging/current_cost�A<����+       ��K	QE��A�*

logging/current_cost��6<²��+       ��K	��E��A�*

logging/current_cost��,<r��,+       ��K	�E��A�*

logging/current_cost�r#<�=#+       ��K	��E��A�*

logging/current_cost��<���+       ��K	�F��A�*

logging/current_cost
�<�q6+       ��K	�:F��A�*

logging/current_cost<=<�;��+       ��K	�gF��A�*

logging/current_costT<���+       ��K	I�F��A�*

logging/current_cost���;�ͅ�+       ��K	��F��A�*

logging/current_cost5V�;Z�x+       ��K	��F��A�*

logging/current_cost	��;���+       ��K	 G��A�*

logging/current_cost��;Ar��+       ��K	�MG��A�*

logging/current_cost`�;N|��+       ��K	�{G��A�*

logging/current_cost�y�;@��b+       ��K	��G��A�*

logging/current_cost�m�;�T��+       ��K	��G��A�*

logging/current_costX$�;f�A�+       ��K	H��A�*

logging/current_cost]��;&>@G+       ��K	<H��A�*

logging/current_costⳳ;*���+       ��K	jH��A�*

logging/current_cost�c�;�%4�+       ��K	h�H��A�*

logging/current_costӛ�;���+       ��K	N�H��A�*

logging/current_cost�N�;��Ǔ+       ��K	�I��A�*

logging/current_cost�l�;���>+       ��K	�.I��A�*

logging/current_cost��;6�fs+       ��K	�\I��A�*

logging/current_costʶ�;���+       ��K	{�I��A�*

logging/current_cost�О;�kj+       ��K	d�I��A�*

logging/current_costo/�;��^�+       ��K	e�I��A�*

logging/current_cost�ɛ;#�Y+       ��K	�J��A�*

logging/current_cost���;y�w+       ��K	�OJ��A�*

logging/current_cost���;��p+       ��K	r~J��A�*

logging/current_cost���;���+       ��K	�J��A�*

logging/current_cost�	�;?�7+       ��K	��J��A�*

logging/current_cost�k�;����+       ��K	�K��A�*

logging/current_cost��;F�k�+       ��K	�:K��A�*

logging/current_cost�j�;�6W�+       ��K	GgK��A�*

logging/current_cost��;���8+       ��K	֖K��A�*

logging/current_costY��;Q��+       ��K	��K��A�*

logging/current_cost!T�;|J{S+       ��K	��K��A�*

logging/current_cost	�;zv�+       ��K	g#L��A�*

logging/current_costÔ;cw�+       ��K	�QL��A�*

logging/current_cost���;�~+       ��K	�~L��A�*

logging/current_cost�F�;Џ��+       ��K	ƬL��A�*

logging/current_cost�;We��+       ��K	O�L��A�*

logging/current_cost�ړ;4p�R+       ��K	�M��A�*

logging/current_cost��;	>��+       ��K	�6M��A�*

logging/current_cost5z�;�;��+       ��K	eeM��A�*

logging/current_cost�L�;;��2+       ��K	ҖM��A�*

logging/current_cost� �;[v�U+       ��K	��M��A�*

logging/current_costd��;���N+       ��K	C�M��A�*

logging/current_cost{ɒ;����+       ��K	�'N��A�*

logging/current_costD��;ź��+       ��K	�VN��A�*

logging/current_cost�s�;�9�+       ��K	�N��A�*

logging/current_cost?I�;h�,n+       ��K	��N��A�*

logging/current_cost��;k�5+       ��K	��N��A�*

logging/current_costI��;���T+       ��K	�O��A�*

logging/current_cost�̑;�ܐ�+       ��K	�CO��A�*

logging/current_costϤ�;��F�+       ��K	`�O��A�*

logging/current_cost�}�;�k�+       ��K	T�O��A�*

logging/current_costW�;�=-�+       ��K	�P��A�*

logging/current_cost�1�;���+       ��K	IP��A�*

logging/current_cost�;΂(�+       ��K	ÇP��A�*

logging/current_cost-�;Sm+       ��K	ԼP��A�*

logging/current_cost'Ȑ;p�g�+       ��K	?�P��A�*

logging/current_costߦ�;�%d+       ��K	�4Q��A�*

logging/current_cost;�	c6+       ��K	�lQ��A�*

logging/current_cost6d�;�6+       ��K	��Q��A�*

logging/current_costY?�;2�z8+       ��K	A�Q��A�*

logging/current_cost��;A�d+       ��K	�R��A�*

logging/current_cost��;�V��+       ��K	�IR��A�*

logging/current_cost�ŏ;ŕ�+       ��K	�{R��A�*

logging/current_costv��;�E�+       ��K	D�R��A�*

logging/current_cost�r�;�Ev(+       ��K	E�R��A�*

logging/current_costJ�;�Dv+       ��K	S��A�*

logging/current_cost"�;��E�+       ��K	BS��A�*

logging/current_cost���;�Q�)+       ��K	�wS��A�*

logging/current_costՎ;iO�+       ��K	j�S��A�*

logging/current_cost��;XM�B+       ��K	H�S��A�*

logging/current_costr��;�@��+       ��K	�T��A�*

logging/current_costBg�;�b>+       ��K	rHT��A�*

logging/current_cost�C�;�Cv�+       ��K	u|T��A�*

logging/current_cost� �;����+       ��K	��T��A�*

logging/current_cost%��;2���+       ��K	��T��A�*

logging/current_costWۍ;aR��+       ��K	�U��A�*

logging/current_cost/��;*�P�+       ��K	BU��A�*

logging/current_costj��;�o�i+       ��K	QqU��A�*

logging/current_costx�;f�A_+       ��K	��U��A�*

logging/current_cost�W�;�Ӏo+       ��K	R�U��A�*

logging/current_cost8�;6_0Y+       ��K	V��A�*

logging/current_cost�;5�Ty+       ��K	�JV��A�*

logging/current_cost���;A֕�+       ��K	�zV��A�*

logging/current_cost8܌;bn+       ��K	�V��A�*

logging/current_cost���;��j�+       ��K	s�V��A�*

logging/current_costj��;��;+       ��K	�W��A�*

logging/current_costЀ�;�U�+       ��K	=W��A�*

logging/current_cost�b�;8�g�+       ��K	olW��A�*

logging/current_costD�;����+       ��K	��W��A�*

logging/current_cost�&�;�O��+       ��K	u�W��A�*

logging/current_cost��;[P�+       ��K	�X��A�*

logging/current_cost��;4L��+       ��K	6/X��A�*

logging/current_cost�͋;r��L+       ��K	�^X��A�*

logging/current_cost��;����+       ��K	��X��A�*

logging/current_costǒ�;=�� +       ��K	ĻX��A�*

logging/current_costxu�;�RH^+       ��K	��X��A�*

logging/current_cost�X�;zW�j+       ��K	Y��A�*

logging/current_costY<�;&B��+       ��K	�LY��A�*

logging/current_cost� �;�Ti+       ��K	�yY��A�*

logging/current_cost��;ʍ��+       ��K	q�Y��A�*

logging/current_cost��;^��+       ��K	��Y��A�*

logging/current_cost�ڊ;/�+       ��K	�	Z��A�*

logging/current_cost�Ȋ;0_+       ��K	�8Z��A�*

logging/current_cost���;S�-�+       ��K	�eZ��A�*

logging/current_cost���;�� R+       ��K	�Z��A�*

logging/current_cost^��;е$�+       ��K	��Z��A�*

logging/current_costP��;=�U+       ��K	`�Z��A�*

logging/current_costu�;��.^+       ��K	,[��A�*

logging/current_costKf�;�o��+       ��K	
Z[��A�*

logging/current_cost�W�;[X�+       ��K	A�[��A�*

logging/current_costI�;��2l+       ��K	\�[��A�*

logging/current_cost�:�;��/�+       ��K	=�[��A�*

logging/current_cost,�;)���+       ��K	3\��A�*

logging/current_cost�;���v+       ��K	�G\��A�*

logging/current_costZ�;�T�~+       ��K	�x\��A�*

logging/current_costW��;l�n�+       ��K	1�\��A�*

logging/current_cost��;z:p�+       ��K	7�\��A�*

logging/current_cost�ԉ;�	+       ��K	�]��A�*

logging/current_cost���;�o5+       ��K	b8]��A�*

logging/current_cost��;WL9�+       ��K	�f]��A�*

logging/current_cost��;(�Y�+       ��K	��]��A�*

logging/current_cost���;��[+       ��K	4�]��A�*

logging/current_cost�u�;O�-+       ��K	��]��A�*

logging/current_costwd�;��x,+       ��K	~!^��A�*

logging/current_cost�S�;Hq +       ��K	kS^��A�*

logging/current_costC�;|8&p+       ��K	�^��A�*

logging/current_cost�6�;ْ�+       ��K	v�^��A�*

logging/current_costE,�;N�+       ��K	��^��A�*

logging/current_cost�!�;��
X+       ��K	Z_��A�*

logging/current_cost_�;�06+       ��K	ZD_��A�*

logging/current_cost��;TA��+       ��K	t_��A�*

logging/current_cost �;ȫ%j+       ��K	Π_��A�*

logging/current_cost���;[�+       ��K	
�_��A�*

logging/current_cost��;�Rf@+       ��K	f�_��A�*

logging/current_cost#�;b~��+       ��K	�.`��A�*

logging/current_cost���;���+       ��K	t]`��A�*

logging/current_costY؈;mG|+       ��K	�`��A�*

logging/current_costCЈ;�A�+       ��K	��`��A�*

logging/current_cost�Ȉ;=�+L+       ��K	��`��A�*

logging/current_cost8��;C��N+       ��K	�a��A�*

logging/current_cost鹈;)Y_
+       ��K	�Ia��A�*

logging/current_cost���;�n�+       ��K	#wa��A�*

logging/current_costw��;l���+       ��K	Ǥa��A�*

logging/current_cost���;�
g�+       ��K	��a��A�*

logging/current_cost��;�/g+       ��K	��a��A�*

logging/current_cost���;<0R�+       ��K	�,b��A�*

logging/current_cost]��;����+       ��K	�[b��A�*

logging/current_cost鋈;�D+       ��K	ʊb��A�*

logging/current_cost���;ls
�+       ��K	�b��A�*

logging/current_costȁ�;�O+       ��K	�b��A�*

logging/current_cost}�;m;�3+       ��K	3c��A�*

logging/current_cost$x�;F�}+       ��K	vCc��A�*

logging/current_cost�s�;�|+       ��K	Yqc��A�*

logging/current_cost�o�;����+       ��K	A�c��A�*

logging/current_cost�k�;�8+       ��K	1�c��A�*

logging/current_cost�g�;|(�+       ��K	9�c��A�*

logging/current_cost�c�;�|=+       ��K	(d��A�*

logging/current_cost%`�;w*�D+       ��K	�Vd��A�*

logging/current_cost\�;�9��+       ��K	0�d��A�*

logging/current_cost/Y�;R<�+       ��K	=�d��A�*

logging/current_cost�U�; �?o+       ��K	&�d��A�*

logging/current_cost�R�;�ڕ+       ��K	be��A�*

logging/current_cost�N�;7�+       ��K	<e��A�*

logging/current_cost�K�;DmQ�+       ��K	vie��A�*

logging/current_costI�;kZ�+       ��K	�e��A�*

logging/current_costF�;|e��+       ��K	��e��A�*

logging/current_cost<C�;	c�+       ��K	&�e��A�*

logging/current_cost?@�;�a�+       ��K	!f��A�*

logging/current_cost1=�;O|{�+       ��K	�Nf��A�*

logging/current_cost9:�;(ga+       ��K	�|f��A�*

logging/current_cost!7�;۱�+       ��K	Īf��A�*

logging/current_cost4�;�G�+       ��K	�f��A�*

logging/current_cost�0�;Vj��+       ��K	�g��A�*

logging/current_cost�,�;��(f+       ��K	�:g��A�*

logging/current_cost�(�;���+       ��K	�fg��A�*

logging/current_costy$�;��� +       ��K	��g��A�*

logging/current_cost8 �;�:�Y+       ��K	+�g��A�*

logging/current_cost��;��څ+       ��K	uh��A�*

logging/current_cost��;1G�+       ��K	�2h��A�*

logging/current_cost��;��!+       ��K	%`h��A�*

logging/current_cost��;^P��+       ��K	U�h��A�*

logging/current_cost��;Lg��+       ��K	H�h��A�*

logging/current_cost��;.>��+       ��K	x�h��A�*

logging/current_cost���;�{�!+       ��K	�i��A�*

logging/current_cost��;蹶+       ��K	�Gi��A�*

logging/current_costG��;+�q3+       ��K	�ui��A�*

logging/current_cost���;\��|+       ��K	��i��A�*

logging/current_cost0�;E|=+       ��K	,�i��A�*

logging/current_cost��;�w��+       ��K	� j��A�*

logging/current_cost��;���+       ��K	�/j��A�*

logging/current_cost�߇;	e�+       ��K	^j��A�*

logging/current_cost܇;Y2�p+       ��K	�j��A�*

logging/current_costU؇;��O<+       ��K	��j��A�*

logging/current_cost�ԇ;���=+       ��K	B�j��A�*

logging/current_costEч;H�+       ��K	�k��A�*

logging/current_cost�͇;K���+       ��K	}Fk��A�*

logging/current_cost�ɇ;��q�+       ��K	�wk��A�*

logging/current_cost�Ň;���+       ��K	��k��A�*

logging/current_cost#;��S+       ��K	��k��A�*

logging/current_cost���;�^��+       ��K	�l��A�	*

logging/current_cost캇;N��+       ��K	�6l��A�	*

logging/current_costY��;K��+       ��K	!il��A�	*

logging/current_costⳇ;ߞ�+       ��K	i�l��A�	*

logging/current_costN��;�V`+       ��K	y�l��A�	*

logging/current_cost߬�;��"�+       ��K	�l��A�	*

logging/current_costQ��;���D+       ��K	Z$m��A�	*

logging/current_cost���;�l��+       ��K	�Rm��A�	*

logging/current_cost���;L��+       ��K	�m��A�	*

logging/current_cost9��;�ޛ�+       ��K	�m��A�	*

logging/current_cost���;�;#+       ��K	��m��A�	*

logging/current_costƘ�;��p]+       ��K	�n��A�	*

logging/current_cost���;��e[+       ��K	�Bn��A�	*

logging/current_cost���;��+       ��K	�qn��A�	*

logging/current_costh��;8?�+       ��K	�n��A�	*

logging/current_costO��;bczg+       ��K	C�n��A�	*

logging/current_cost?��;��+       ��K	w�n��A�	*

logging/current_cost0��;�*+       ��K	 ,o��A�	*

logging/current_cost!��;nB�	+       ��K	v]o��A�	*

logging/current_cost��;Qn;�+       ��K	��o��A�	*

logging/current_costl|�;���+       ��K	I�o��A�	*

logging/current_cost�x�;7��+       ��K	O�o��A�	*

logging/current_cost�u�;�\o+       ��K	tp��A�	*

logging/current_costAr�;T�+       ��K	Fp��A�	*

logging/current_cost>o�;�g��+       ��K	�up��A�	*

logging/current_cost>l�;_��K+       ��K	6�p��A�
*

logging/current_cost8i�;Mo�m+       ��K	��p��A�
*

logging/current_cost�f�;cN�+       ��K	��p��A�
*

logging/current_costd�;�ʬ+       ��K	-q��A�
*

logging/current_cost�a�;*��+       ��K	�Zq��A�
*

logging/current_cost'_�;��+       ��K	�q��A�
*

logging/current_cost�\�;}�~+       ��K	�q��A�
*

logging/current_costrZ�;����+       ��K	4�q��A�
*

logging/current_cost*X�;�k%+       ��K	�r��A�
*

logging/current_cost�U�;>Pz�+       ��K	y@r��A�
*

logging/current_cost�S�;)��+       ��K	�mr��A�
*

logging/current_costoQ�;.i� +       ��K	ȝr��A�
*

logging/current_costIO�;�a0+       ��K	��r��A�
*

logging/current_costM�;��6+       ��K	B�r��A�
*

logging/current_cost�J�;4�7�+       ��K	�#s��A�
*

logging/current_cost�H�;��̱+       ��K	sRs��A�
*

logging/current_cost�F�;ȳ�+       ��K	g�s��A�
*

logging/current_cost�D�;��+       ��K	��s��A�
*

logging/current_cost�B�;!
(�+       ��K	��s��A�
*

logging/current_costq@�;��J�+       ��K	Et��A�
*

logging/current_costb>�;<M'�+       ��K	�=t��A�
*

logging/current_costU<�;�r�^+       ��K	�mt��A�
*

logging/current_costH:�;���+       ��K	��t��A�
*

logging/current_cost=8�;Ҳ�g+       ��K	��t��A�
*

logging/current_cost76�;6*]+       ��K	�u��A�
*

logging/current_cost-4�;�tz�+       ��K	e/u��A�
*

logging/current_cost'2�;�]��+       ��K	_u��A�*

logging/current_cost$0�;g���+       ��K	��u��A�*

logging/current_cost .�;��F�+       ��K	��u��A�*

logging/current_cost,�;ј$+       ��K	�u��A�*

logging/current_cost�)�;�,Q)+       ��K	hv��A�*

logging/current_cost�'�;����+       ��K	�Av��A�*

logging/current_cost%�;J�3�+       ��K	�qv��A�*

logging/current_cost�"�;wFW�+       ��K	P�v��A�*

logging/current_cost2 �;qZ�+       ��K	��v��A�*

logging/current_cost��;�2��+       ��K	d�v��A�*

logging/current_costB�;�x)+       ��K	�*w��A�*

logging/current_cost��;)��+       ��K	�Ww��A�*

logging/current_costT�;��n+       ��K	��w��A�*

logging/current_cost��;��T+       ��K	9�w��A�*

logging/current_costo�;�1i�+       ��K	�w��A�*

logging/current_cost �;t���+       ��K	1x��A�*

logging/current_cost��;��
V+       ��K	MEx��A�*

logging/current_cost%
�;���+       ��K	�qx��A�*

logging/current_cost��;Kl S+       ��K	�x��A�*

logging/current_costP�;`=2+       ��K	��x��A�*

logging/current_cost��;�j�-+       ��K	�y��A�*

logging/current_cost� �;��+       ��K	�-y��A�*

logging/current_cost'��;�2��+       ��K	�^y��A�*

logging/current_cost���;���~+       ��K	̋y��A�*

logging/current_costr��;��T�+       ��K	
�y��A�*

logging/current_cost��;^��+       ��K	��y��A�*

logging/current_cost��;ʷ�n+       ��K	z��A�*

logging/current_costt�;?�3+       ��K	JEz��A�*

logging/current_cost$��;�d�?+       ��K	isz��A�*

logging/current_cost��;��T+       ��K	"�z��A�*

logging/current_cost��;Lc�+       ��K	�z��A�*

logging/current_cost>�;޾��+       ��K	4{��A�*

logging/current_cost��;c;��+       ��K	12{��A�*

logging/current_cost��;�@�+       ��K	�`{��A�*

logging/current_costl�;���+       ��K	��{��A�*

logging/current_cost+��;S�~�+       ��K	v�{��A�*

logging/current_cost�݆;p*+       ��K	1|��A�*

logging/current_cost�ۆ;Jꐲ+       ��K	ph|��A�*

logging/current_costxن;e-5�+       ��K	�|��A�*

logging/current_cost?׆;@��p+       ��K	��|��A�*

logging/current_costՆ;���+       ��K	}��A�*

logging/current_cost�҆;Ԛ� +       ��K	�S}��A�*

logging/current_cost�І;"��+       ��K	�}��A�*

logging/current_costvΆ;$Lň+       ��K	�}��A�*

logging/current_costĬ;J�+       ��K	�	~��A�*

logging/current_cost!ʆ;=�+       ��K	3A~��A�*

logging/current_cost�ǆ;NL��+       ��K	�p~��A�*

logging/current_cost�ņ;�e�+       ��K	��~��A�*

logging/current_cost�Æ;㏽�+       ��K	��~��A�*

logging/current_cost���;tK)+       ��K	
��A�*

logging/current_costr��;X�!J+       ��K	r6��A�*

logging/current_costW��;Ų��+       ��K	�g��A�*

logging/current_cost<��;�м�+       ��K	���A�*

logging/current_cost"��;���+       ��K	����A�*

logging/current_cost��;�lj�+       ��K	����A�*

logging/current_cost���;:�a+       ��K	{'���A�*

logging/current_cost겆;+0�+       ��K	LV���A�*

logging/current_cost۰�;���+       ��K	�����A�*

logging/current_costϮ�;�~�+       ��K	C����A�*

logging/current_costƬ�;�D��+       ��K	|瀷�A�*

logging/current_cost���;���|+       ��K	����A�*

logging/current_cost���;%��+       ��K	H���A�*

logging/current_cost���;����+       ��K	v���A�*

logging/current_cost���;,\h�+       ��K	����A�*

logging/current_cost���;ht�~+       ��K	�Ӂ��A�*

logging/current_cost���;���+       ��K	����A�*

logging/current_costŞ�;���+       ��K	�8���A�*

logging/current_costќ�;G��\+       ��K	wn���A�*

logging/current_costښ�;(}h+       ��K	I����A�*

logging/current_cost阆;�Ha+       ��K	�ӂ��A�*

logging/current_cost���;Kȁ�+       ��K	V���A�*

logging/current_cost��;�.r+       ��K	3���A�*

logging/current_cost"��;f��+       ��K	?n���A�*

logging/current_cost;��;�_n
+       ��K	­���A�*

logging/current_costW��;���+       ��K	�݃��A�*

logging/current_costs��;��J+       ��K	����A�*

logging/current_cost���;h���+       ��K	�E���A�*

logging/current_cost���;��Ԗ+       ��K	�v���A�*

logging/current_costڇ�;<��;+       ��K	�����A�*

logging/current_cost ��;l��+       ��K	߄��A�*

logging/current_cost+��;e
#+       ��K	����A�*

logging/current_costW��;����+       ��K	D���A�*

logging/current_cost���;�h�^+       ��K	�t���A�*

logging/current_cost�~�;�&�r+       ��K	G����A�*

logging/current_cost�|�;�� +       ��K	�ޅ��A�*

logging/current_cost {�;'Ύ�+       ��K	f���A�*

logging/current_costYy�;����+       ��K	�B���A�*

logging/current_cost�w�;+P�+       ��K	�u���A�*

logging/current_cost�u�;u,a+       ��K	I����A�*

logging/current_costt�;���+       ��K	�↷�A�*

logging/current_costVr�;���+       ��K	,*���A�*

logging/current_cost�p�;��5+       ��K	n���A�*

logging/current_cost�n�;����+       ��K	�ˇ��A�*

logging/current_cost-m�;�0]3+       ��K	_���A�*

logging/current_cost}k�;�lLr+       ��K	�3���A�*

logging/current_cost�i�;��~+       ��K	wi���A�*

logging/current_cost"h�;UN[�+       ��K	�����A�*

logging/current_costvf�;�Y+       ��K	XЈ��A�*

logging/current_cost�d�;v��+       ��K	l.���A�*

logging/current_cost)c�;����+       ��K	�x���A�*

logging/current_cost�a�;%Wά+       ��K	+ŉ��A�*

logging/current_cost�_�;:0�O+       ��K	����A�*

logging/current_costE^�;>�	+       ��K	�X���A�*

logging/current_cost�\�;����+       ��K	C����A�*

logging/current_cost[�;9�|"+       ��K	�����A�*

logging/current_costzY�;���+       ��K	�+���A�*

logging/current_cost�W�;E���+       ��K	�Z���A�*

logging/current_costUV�;B2#�+       ��K	�����A�*

logging/current_cost�T�;���+       ��K	�΋��A�*

logging/current_cost:S�;�N��+       ��K	����A�*

logging/current_cost�Q�;�U��+       ��K	!H���A�*

logging/current_cost(P�;��J+       ��K	�}���A�*

logging/current_cost�N�;��#+       ��K	=����A�*

logging/current_cost$M�;��Q�+       ��K	`挷�A�*

logging/current_cost�K�;�Rh+       ��K	:���A�*

logging/current_cost%J�;�RU�+       ��K	MR���A�*

logging/current_cost�H�;���+       ��K	G����A�*

logging/current_cost2G�;8�+       ��K	Է���A�*

logging/current_cost�E�;O�+       ��K	3퍷�A�*

logging/current_costJD�;J�WX+       ��K	�.���A�*

logging/current_cost�B�;��rl+       ��K	�j���A�*

logging/current_costuA�;�I�+       ��K	;����A�*

logging/current_cost
@�;��z{+       ��K	�ʎ��A�*

logging/current_cost�>�;ٔ�+       ��K	�����A�*

logging/current_costD=�;���+       ��K	t-���A�*

logging/current_cost�;�;�+i+       ��K	!i���A�*

logging/current_cost:�;)`{+       ��K	�����A�*

logging/current_cost!9�;V�}�+       ��K	ۏ��A�*

logging/current_cost�7�;����+       ��K	�
���A�*

logging/current_costu6�;F0:�+       ��K	�8���A�*

logging/current_cost!5�; �dx+       ��K	�q���A�*

logging/current_cost�3�;.J9�+       ��K	7����A�*

logging/current_cost�2�;b�T:+       ��K	xҐ��A�*

logging/current_cost61�;8i�+       ��K	@���A�*

logging/current_cost�/�; )&�+       ��K	5���A�*

logging/current_cost�.�;UN��+       ��K	�h���A�*

logging/current_costb-�;�X9+       ��K	�����A�*

logging/current_cost&,�;5`��+       ��K	�ˑ��A�*

logging/current_cost�*�;��%�+       ��K	�����A�*

logging/current_cost�)�;@�+       ��K	k,���A�*

logging/current_costn(�;q�L+       ��K	�\���A�*

logging/current_cost9'�;:�p+       ��K	痒��A�*

logging/current_cost&�;�}J7+       ��K	�ǒ��A�*

logging/current_costX$�;�j�+       ��K	P����A�*

logging/current_cost"�;b�*1+       ��K	�(���A�*

logging/current_cost��;x�++       ��K	3X���A�*

logging/current_cost�;";��+       ��K	u����A�*

logging/current_costy�;KLs+       ��K	f��A�*

logging/current_cost��;�՜I+       ��K	c����A�*

logging/current_cost/�;�W��+       ��K	.���A�*

logging/current_cost��;J�5�+       ��K	�^���A�*

logging/current_cost��;�V^t+       ��K	�����A�*

logging/current_costY�;!h�+       ��K	 Δ��A�*

logging/current_cost�
�;>)��+       ��K	����A�*

logging/current_cost/�;�3=+       ��K	�/���A�*

logging/current_cost��;t9�+       ��K	`���A�*

logging/current_cost#�;'�+       ��K	؎���A�*

logging/current_cost� �;wѨ�+       ��K	Bѕ��A�*

logging/current_cost*��;�ٔ_+       ��K	)���A�*

logging/current_cost���;�� +       ��K	7���A�*

logging/current_costL��;Q7(+       ��K	�h���A�*

logging/current_cost���;I�+       ��K	�����A�*

logging/current_costx�;I��+       ��K	=͖��A�*

logging/current_cost�;(S��+       ��K	�����A�*

logging/current_cost��;X�i�+       ��K	�,���A�*

logging/current_costf�;�;+       ��K	o[���A�*

logging/current_cost�;r��+       ��K	ً���A�*

logging/current_cost��;��|�+       ��K	]����A�*

logging/current_costx�;7��+       ��K	���A�*

logging/current_cost3�;^_�+       ��K	P!���A�*

logging/current_cost��;D_��+       ��K	O���A�*

logging/current_cost�߅;[�K�+       ��K	�{���A�*

logging/current_costz݅;*��+       ��K	ߪ���A�*

logging/current_costEۅ;��T�+       ��K	�ט��A�*

logging/current_costم;+��+       ��K		���A�*

logging/current_cost�օ;g��+       ��K	8���A�*

logging/current_cost�ԅ;��+       ��K	dh���A�*

logging/current_cost�҅;�6�+       ��K	Ε���A�*

logging/current_costЅ;S���+       ��K		ř��A�*

logging/current_costd΅;��;�+       ��K	���A�*

logging/current_costM̅;7]]�+       ��K	
%���A�*

logging/current_cost:ʅ;�-�;+       ��K	8Y���A�*

logging/current_cost+ȅ;/}��+       ��K	6����A�*

logging/current_cost"ƅ;|���+       ��K	G����A�*

logging/current_costą;�~N+       ��K	5ꚷ�A�*

logging/current_cost;{[M�+       ��K	����A�*

logging/current_cost��;a�.�+       ��K	�I���A�*

logging/current_cost��;����+       ��K	�z���A�*

logging/current_cost)��;�(�+       ��K	����A�*

logging/current_cost5��;!2�*+       ��K	�֛��A�*

logging/current_costJ��;V�`�+       ��K	m	���A�*

logging/current_cost]��;+�ž+       ��K	�9���A�*

logging/current_costw��;��ξ+       ��K	�h���A�*

logging/current_cost���;�!=�+       ��K	�����A�*

logging/current_cost���;���+       ��K	Ŝ��A�*

logging/current_costϮ�;�g�W+       ��K	 ����A�*

logging/current_cost�;�i+       ��K	v"���A�*

logging/current_cost��;9�)�+       ��K	'R���A�*

logging/current_cost=��;a���+       ��K	˅���A�*

logging/current_costh��;����+       ��K	����A�*

logging/current_cost���;SC�M+       ��K	�杷�A�*

logging/current_costţ�;�$�+       ��K	N���A�*

logging/current_cost���;�=�+       ��K	�D���A�*

logging/current_cost6��;� �+       ��K	�r���A�*

logging/current_costu��;�(�(+       ��K	;����A�*

logging/current_cost���;�w�!+       ��K	�̞��A�*

logging/current_cost��;�Dh,+       ��K	�����A�*

logging/current_costK��;$���+       ��K	 ,���A�*

logging/current_cost���;MB�+       ��K	HX���A�*

logging/current_cost^��;'���+       ��K	P����A�*

logging/current_cost�;��Ǻ+       ��K	P����A�*

logging/current_costc��;{
�+       ��K	�䟷�A�*

logging/current_costЍ�;��ݘ+       ��K	����A�*

logging/current_cost6��;�Z��+       ��K	�A���A�*

logging/current_cost���;�+(]+       ��K	p���A�*

logging/current_cost ��;(��+       ��K	Z����A�*

logging/current_costi��;���+       ��K	�͠��A�*

logging/current_costՀ�;`��Y+       ��K	�����A�*

logging/current_costD~�;m֊n+       ��K	�*���A�*

logging/current_cost�{�;4���+       ��K	�V���A�*

logging/current_cost5y�;�*�+       ��K	�����A�*

logging/current_cost�v�;����+       ��K	ൡ��A�*

logging/current_cost8t�;��w�+       ��K	@桷�A�*

logging/current_cost�q�;�
"�+       ��K	����A�*

logging/current_costMo�;���+       ��K	�B���A�*

logging/current_cost�l�;�I4P+       ��K	�r���A�*

logging/current_cost�k�;Ӳ��+       ��K	~����A�*

logging/current_cost3k�;�.�+       ��K	$Т��A�*

logging/current_cost�i�;�+v+       ��K	�����A�*

logging/current_cost�i�;g��@+       ��K	,���A�*

logging/current_cost�h�;lY�+       ��K	\���A�*

logging/current_costZh�;��.u+       ��K	ǋ���A�*

logging/current_cost�g�;c��v+       ��K	����A�*

logging/current_cost�g�;��0+       ��K	�磷�A�*

logging/current_cost)g�;�Ɨ+       ��K	E���A�*

logging/current_cost�f�;b��V+       ��K	]C���A�*

logging/current_cost�f�;c��+       ��K	2s���A�*

logging/current_cost4f�;�6�+       ��K	�����A�*

logging/current_cost�e�;��Ѕ+       ��K	�Ϥ��A�*

logging/current_cost�e�;�ڢe+       ��K	h����A�*

logging/current_costCe�;v0ŵ+       ��K	".���A�*

logging/current_coste�;	�}�+       ��K	�_���A�*

logging/current_cost�d�;�6�r+       ��K	-����A�*

logging/current_costnd�;���+       ��K	)����A�*

logging/current_cost$d�;}6I+       ��K	h楷�A�*

logging/current_cost�c�;�H�+       ��K	Y���A�*

logging/current_cost�c�;���+       ��K	B@���A�*

logging/current_costnc�;9t�+       ��K	p���A�*

logging/current_costc�;G{��+       ��K	R����A�*

logging/current_cost�b�;�.�+       ��K	�ʦ��A�*

logging/current_cost�b�;R�iF+       ��K	�����A�*

logging/current_cost^b�;�_.+       ��K	�,���A�*

logging/current_cost$b�;�¸�+       ��K	�Z���A�*

logging/current_cost�a�;渒+       ��K	����A�*

logging/current_cost�a�;��}+       ��K	����A�*

logging/current_cost�a�;!Lp)+       ��K	d觷�A�*

logging/current_cost@a�;�E++       ��K	g���A�*

logging/current_cost a�;:�q�+       ��K	NC���A�*

logging/current_cost�`�; �g+       ��K	�p���A�*

logging/current_cost�`�;��U�+       ��K	�����A�*

logging/current_costc`�;���k+       ��K	�ͨ��A�*

logging/current_cost!`�;l��+       ��K	����A�*

logging/current_cost�_�;����+       ��K	�1���A�*

logging/current_cost�_�;��/+       ��K	I_���A�*

logging/current_costs_�;�^�+       ��K	�����A�*

logging/current_costD_�;���+       ��K	C����A�*

logging/current_cost_�;�|;+       ��K	��A�*

logging/current_cost�^�;��a+       ��K	l���A�*

logging/current_cost�^�;]��+       ��K	�I���A�*

logging/current_cost{^�;�y��+       ��K		x���A�*

logging/current_costH^�;��+       ��K	<����A�*

logging/current_cost^�;2���+       ��K	Iժ��A�*

logging/current_cost�]�;"�7�+       ��K	:���A�*

logging/current_cost�]�;��+       ��K	D:���A�*

logging/current_costk]�;��}+       ��K	ah���A�*

logging/current_cost;]�;��?�+       ��K	����A�*

logging/current_cost]�;i�/+       ��K	�����A�*

logging/current_cost�\�;��+       ��K	���A�*

logging/current_cost�\�;^Q�@+       ��K	6"���A�*

logging/current_cost�\�;u�+       ��K	�Q���A�*

logging/current_costM\�;���+       ��K	����A�*

logging/current_cost\�;']J�+       ��K	i����A�*

logging/current_cost\�;�hơ+       ��K	Rଷ�A�*

logging/current_cost�[�;�L()+       ��K	���A�*

logging/current_cost�[�;��+       ��K	�=���A�*

logging/current_costt[�;����+       ��K	n���A�*

logging/current_costP[�;�?�+       ��K	1����A�*

logging/current_cost[�;�Me�+       ��K	�ȭ��A�*

logging/current_cost�Z�;FI�+       ��K	�����A�*

logging/current_cost�Z�;���+       ��K	�(���A�*

logging/current_cost�Z�;U��+       ��K	>W���A�*

logging/current_costxZ�;���{+       ��K	򃮷�A�*

logging/current_costVZ�;Q<+       ��K	�����A�*

logging/current_cost/Z�;xV��+       ��K	�⮷�A�*

logging/current_cost�Y�;�9�+       ��K	/���A�*

logging/current_cost�Y�;�f_x+       ��K	�B���A�*

logging/current_cost9Y�;�x9+       ��K	�o���A�*

logging/current_cost�X�;G��"+       ��K	�����A�*

logging/current_costHX�;"%�+       ��K	�ȯ��A�*

logging/current_cost�W�;�)	a+       ��K	w����A�*

logging/current_cost6W�;R¯�+       ��K	�#���A�*

logging/current_cost�V�;9|�+       ��K	�Q���A�*

logging/current_costNV�;�ɻ+       ��K	W����A�*

logging/current_cost�U�;q:߾+       ��K	?����A�*

logging/current_cost�U�;��I�+       ��K	.۰��A�*

logging/current_costeU�; �\U+       ��K	2���A�*

logging/current_cost U�;�!+       ��K	n6���A�*

logging/current_cost�T�;�!��+       ��K	�a���A�*

logging/current_cost�T�;����+       ��K	�����A�*

logging/current_cost=T�;>�+       ��K	�����A�*

logging/current_costT�;qk��+       ��K	�����A�*

logging/current_cost�S�;9�+       ��K	����A�*

logging/current_cost}S�;� H�+       ��K	�I���A�*

logging/current_costS�;%�C�+       ��K	Xu���A�*

logging/current_cost�R�;a��+       ��K	�����A�*

logging/current_costoR�;���+       ��K	Lв��A�*

logging/current_cost%R�;�*�>+       ��K	[����A�*

logging/current_cost�Q�;i��Y+       ��K	�,���A�*

logging/current_cost�Q�;�Q�z+       ��K	�Y���A�*

logging/current_costlQ�;/\�M+       ��K	�����A�*

logging/current_costAQ�;p�;q+       ��K	ڵ���A�*

logging/current_costQ�;{��+       ��K	�峷�A�*

logging/current_cost�P�;�:6�+       ��K	o���A�*

logging/current_cost�P�;p�+       ��K	}B���A�*

logging/current_costVP�;&��Q+       ��K	�r���A�*

logging/current_costDP�;ߩ�+       ��K	]����A�*

logging/current_costP�;��R�+       ��K	�δ��A�*

logging/current_cost�O�;�G�5+       ��K	�����A�*

logging/current_cost�O�;ݷ��+       ��K	i)���A�*

logging/current_cost�O�;0�Y+       ��K	�W���A�*

logging/current_costdO�;.>'�+       ��K	����A�*

logging/current_cost[O�;���y+       ��K	5����A�*

logging/current_costKO�;Ĺ��+       ��K	�ᵷ�A�*

logging/current_cost)O�;�$�+       ��K	����A�*

logging/current_cost
O�;��k�+       ��K	�B���A�*

logging/current_cost�N�;W�*�+       ��K	ur���A�*

logging/current_costWN�;��3�+       ��K	Н���A�*

logging/current_cost(N�;T��k+       ��K	�̶��A�*

logging/current_cost�M�;Os�j+       ��K	�����A�*

logging/current_cost�M�;8'	�+       ��K	�)���A�*

logging/current_cost�M�;��+       ��K	�V���A�*

logging/current_cost�M�;Z��+       ��K	m����A�*

logging/current_costXM�;&�|+       ��K	3����A�*

logging/current_costGM�;�4�+       ��K	�混�A�*

logging/current_costM�;9�DD+       ��K	���A�*

logging/current_costM�;XxV�+       ��K	yE���A�*

logging/current_cost�L�;��~+       ��K	�t���A�*

logging/current_cost�L�;QS��+       ��K	棸��A�*

logging/current_cost�L�;K�aJ+       ��K	�ϸ��A�*

logging/current_cost�L�;�<�+       ��K	�����A�*

logging/current_cost|L�;w�2+       ��K	�.���A�*

logging/current_cost_L�;��.+       ��K	�\���A�*

logging/current_costPL�;k�+       ��K	�����A�*

logging/current_cost7L�;%��n+       ��K	񸹷�A�*

logging/current_costL�;���	+       ��K	#湷�A�*

logging/current_costL�;�>Ph+       ��K	����A�*

logging/current_cost�K�;qF\�+       ��K	E?���A�*

logging/current_cost�K�;�{B+       ��K	�q���A�*

logging/current_cost�K�;YB_�+       ��K	֟���A�*

logging/current_cost�K�;+       ��K	Vκ��A�*

logging/current_cost�K�;5+�+       ��K	�����A�*

logging/current_cost�K�;�}f+       ��K	�-���A�*

logging/current_costtK�;���+       ��K	]���A�*

logging/current_costdK�;O$�+       ��K	m����A�*

logging/current_cost_K�;_�n�+       ��K	�����A�*

logging/current_cost>K�;yA#v+       ��K	\3���A�*

logging/current_cost9K�;MM�-+       ��K	�m���A�*

logging/current_cost!K�;�y�+       ��K	�����A�*

logging/current_costK�;�Ky+       ��K	�����A�*

logging/current_costK�;��8+       ��K	�$���A�*

logging/current_cost�J�;_Yh+       ��K	6^���A�*

logging/current_cost�J�;�?+       ��K	�����A�*

logging/current_cost�J�;'xZS+       ��K	�ӽ��A�*

logging/current_cost�J�;իp+       ��K	6
���A�*

logging/current_cost�J�;&۷+       ��K	k;���A�*

logging/current_cost�J�;6�'K+       ��K	
}���A�*

logging/current_cost�J�;:J^�+       ��K	¯���A�*

logging/current_cost�J�;,��+       ��K	�ᾷ�A�*

logging/current_cost�J�;�&�0+       ��K	{���A�*

logging/current_cost�J�;�*�N+       ��K	CB���A�*

logging/current_cost�J�;�ϖf+       ��K	q���A�*

logging/current_cost�J�;ҋ�p+       ��K	�����A�*

logging/current_cost�J�;��rI+       ��K	�Ͽ��A�*

logging/current_costOJ�;�� l+       ��K	����A�*

logging/current_costIJ�;;��+       ��K	�0���A�*

logging/current_costPJ�;���]+       ��K	�]���A�*

logging/current_cost:J�;�Kw+       ��K	�����A�*

logging/current_costBJ�;�E�+       ��K	�����A�*

logging/current_cost<J�;�ԝ�+       ��K	v����A�*

logging/current_cost�I�;VdD+       ��K	~���A�*

logging/current_costJ�;���[+       ��K	KJ���A�*

logging/current_cost�I�;WP��+       ��K	�y���A�*

logging/current_cost�I�;>�"+       ��K	̪���A�*

logging/current_cost�I�;f��2+       ��K	�����A�*

logging/current_cost�I�;=�F�+       ��K	3·�A�*

logging/current_cost�I�;���<+       ��K	�6·�A�*

logging/current_cost�I�;���a+       ��K	lk·�A�*

logging/current_cost�I�;a� \+       ��K	��·�A�*

logging/current_cost�I�;+_6�+       ��K	��·�A�*

logging/current_cost�I�;A�1
+       ��K	�p÷�A�*

logging/current_cost�I�;���R+       ��K	B�÷�A�*

logging/current_cost�I�;��U+       ��K	C�÷�A�*

logging/current_cost�I�;�jj+       ��K	ķ�A�*

logging/current_costyI�;�4�+       ��K	�Rķ�A�*

logging/current_costqI�;K[��+       ��K	(�ķ�A�*

logging/current_cost`I�;�Pz�+       ��K	��ķ�A�*

logging/current_costI�;��Y+       ��K	�
ŷ�A�*

logging/current_costPI�;K�5�+       ��K	zGŷ�A�*

logging/current_costXI�;c�Ak+       ��K	�ŷ�A�*

logging/current_costII�;,�j+       ��K	��ŷ�A�*

logging/current_costQI�;�<�+       ��K	 �ŷ�A�*

logging/current_cost@I�;a]+       ��K	hƷ�A�*

logging/current_cost/I�;��2+       ��K	[SƷ�A�*

logging/current_costII�;;6�S+       ��K	�Ʒ�A�*

logging/current_cost[I�;�b +       ��K	��Ʒ�A�*

logging/current_cost:I�;��w�+       ��K	��Ʒ�A�*

logging/current_cost=I�;�"]�+       ��K	,Ƿ�A�*

logging/current_cost@I�;Kl�$+       ��K	�`Ƿ�A�*

logging/current_costII�;��.�+       ��K	^�Ƿ�A�*

logging/current_cost I�;� �x+       ��K	G�Ƿ�A�*

logging/current_costI�;x-m+       ��K	�ȷ�A�*

logging/current_cost+I�;�ʉc+       ��K	/8ȷ�A�*

logging/current_costI�;�O��+       ��K	�jȷ�A�*

logging/current_costI�;���+       ��K	��ȷ�A�*

logging/current_costI�;�=�|+       ��K		�ȷ�A�*

logging/current_costI�;L���+       ��K	�ɷ�A�*

logging/current_cost�H�;�ʸ�+       ��K	n?ɷ�A�*

logging/current_cost�H�;�E�+       ��K	�zɷ�A�*

logging/current_costI�;?��p+       ��K	&�ɷ�A�*

logging/current_cost�H�;��b+       ��K	_�ɷ�A�*

logging/current_cost�H�;V�A�+       ��K	�	ʷ�A�*

logging/current_cost�H�;K� 3+       ��K	T:ʷ�A�*

logging/current_cost�H�;�f#+       ��K	�kʷ�A�*

logging/current_cost�H�;�d�+       ��K	��ʷ�A�*

logging/current_cost�H�;O��+       ��K	�ʷ�A�*

logging/current_cost�H�;�۶�+       ��K	�ʷ�A�*

logging/current_cost�H�;I��+       ��K	]$˷�A�*

logging/current_cost�H�;�l��+       ��K	6R˷�A�*

logging/current_cost�H�;K��+       ��K	��˷�A�*

logging/current_cost�H�;D��z+       ��K	ϯ˷�A�*

logging/current_cost�H�;�xݩ+       ��K	7�˷�A�*

logging/current_cost�H�;[&pp+       ��K	̷�A�*

logging/current_cost�H�;< �G+       ��K	�L̷�A�*

logging/current_cost�H�;�>�+       ��K	|̷�A�*

logging/current_cost�H�;X4�+       ��K	Ш̷�A�*

logging/current_cost�H�;��a+       ��K	��̷�A�*

logging/current_costqH�;�31A+       ��K	B
ͷ�A�*

logging/current_costiH�;;�+       ��K	s6ͷ�A�*

logging/current_cost\H�;$�7+       ��K	�fͷ�A�*

logging/current_costSH�;�	�u+       ��K	5�ͷ�A�*

logging/current_cost`H�;�mK+       ��K	5�ͷ�A�*

logging/current_costwH�;��Y~+       ��K	[�ͷ�A�*

logging/current_cost�H�;���+       ��K	
&η�A�*

logging/current_costPH�;���+       ��K	B\η�A�*

logging/current_cost?H�;�!+       ��K	]�η�A�*

logging/current_costBH�;�7�+       ��K	S�η�A�*

logging/current_cost@H�;#^�+       ��K	n�η�A�*

logging/current_cost-H�;2+       ��K	�Ϸ�A�*

logging/current_cost7H�;+>�+       ��K	(HϷ�A�*

logging/current_cost%H�;D�&+       ��K	��Ϸ�A�*

logging/current_costHH�;n?.+       ��K	�Ϸ�A�*

logging/current_costH�;�.��+       ��K	��Ϸ�A�*

logging/current_costH�;���+       ��K	�!з�A�*

logging/current_cost6H�;��++       ��K	@Oз�A�*

logging/current_costH�;��M=+       ��K	��з�A�*

logging/current_cost9H�;��%�+       ��K	۲з�A�*

logging/current_costH�;/�`$+       ��K	�з�A�*

logging/current_costH�;gg�+       ��K	1ѷ�A�*

logging/current_costH�;�m�"+       ��K	0@ѷ�A�*

logging/current_cost�G�;���c+       ��K	pѷ�A�*

logging/current_costH�;���$+       ��K	r�ѷ�A�*

logging/current_cost�G�;���g+       ��K	��ѷ�A�*

logging/current_costH�;Ӯ*)+       ��K	q ҷ�A�*

logging/current_cost�G�;�� +       ��K	�1ҷ�A�*

logging/current_cost�G�;��1�+       ��K	�fҷ�A�*

logging/current_costH�;`�+       ��K	2�ҷ�A�*

logging/current_cost�G�;FF�+       ��K	��ҷ�A�*

logging/current_cost�G�;k��+       ��K	ӷ�A�*

logging/current_cost�G�;/�A+       ��K	#Nӷ�A�*

logging/current_cost�G�;�U�+       ��K	|ӷ�A�*

logging/current_cost�G�;Vf�W+       ��K	Y�ӷ�A�*

logging/current_cost�G�;e\+       ��K	�ӷ�A�*

logging/current_costH�;��w%+       ��K	  Է�A�*

logging/current_cost�G�;��+       ��K	TԷ�A�*

logging/current_cost�G�;e���+       ��K	��Է�A�*

logging/current_cost�G�;T���+       ��K	�Է�A�*

logging/current_cost�G�;I��+       ��K	��Է�A�*

logging/current_cost�G�;��@c+       ��K	�շ�A�*

logging/current_cost�G�;�b}�+       ��K	�Nշ�A�*

logging/current_cost�G�;�V�+       ��K	�}շ�A�*

logging/current_cost�G�;�D3)+       ��K	*�շ�A�*

logging/current_cost�G�;��J�+       ��K	>�շ�A�*

logging/current_cost�G�;��ZV+       ��K	�ַ�A�*

logging/current_cost�G�;�.O+       ��K	tJַ�A�*

logging/current_cost�G�;4H��+       ��K	�{ַ�A�*

logging/current_cost�G�;���+       ��K	�ַ�A�*

logging/current_cost�G�;�ݠ<+       ��K	��ַ�A�*

logging/current_cost�G�;���m+       ��K	Y׷�A�*

logging/current_cost�G�;��+       ��K	�V׷�A�*

logging/current_cost�G�;B.�+       ��K	 �׷�A�*

logging/current_cost�G�;�K��+       ��K	}�׷�A�*

logging/current_cost�G�;@��+       ��K	_�׷�A�*

logging/current_cost�G�;�M!+       ��K	�ط�A�*

logging/current_cost�G�;�'_5+       ��K	�Kط�A�*

logging/current_cost�G�;�5��+       ��K	�}ط�A�*

logging/current_cost�G�;��2+       ��K	�ط�A�*

logging/current_costG�;��;+       ��K	��ط�A�*

logging/current_cost�G�;��	J+       ��K	0ٷ�A�*

logging/current_costvG�;h���+       ��K	�Mٷ�A�*

logging/current_costmG�;�4�+       ��K	]�ٷ�A�*

logging/current_cost�G�;�HH+       ��K	S�ٷ�A�*

logging/current_costvG�;� +       ��K	.�ٷ�A�*

logging/current_cost�G�;�W�+       ��K	S
ڷ�A�*

logging/current_costuG�;��S�+       ��K	�>ڷ�A�*

logging/current_costbG�;5��G+       ��K	�oڷ�A�*

logging/current_cost�G�;Wt�+       ��K	a�ڷ�A�*

logging/current_costRG�;��Y+       ��K	��ڷ�A�*

logging/current_costrG�;̻װ+       ��K	n�ڷ�A�*

logging/current_costbG�;����+       ��K	.۷�A�*

logging/current_costWG�;�N�(+       ��K	�c۷�A�*

logging/current_costYG�;��I+       ��K	ב۷�A�*

logging/current_costNG�;��^+       ��K	?�۷�A�*

logging/current_costMG�;R�Q+       ��K	J�۷�A�*

logging/current_cost\G�;��?+       ��K	�ܷ�A�*

logging/current_costTG�;�|�+       ��K	xNܷ�A�*

logging/current_costIG�;*� �+       ��K	Hܷ�A�*

logging/current_costQG�;�ڳP+       ��K	��ܷ�A�*

logging/current_costkG�;c'�5+       ��K	O�ܷ�A�*

logging/current_costAG�;�\�+       ��K	�ݷ�A�*

logging/current_cost7G�;(#��+       ��K	iMݷ�A�*

logging/current_costmG�;v%z�+       ��K	�~ݷ�A�*

logging/current_cost?G�;6f]�+       ��K	�ݷ�A�*

logging/current_costPG�;�|ǥ+       ��K	(�ݷ�A�*

logging/current_cost5G�;��+       ��K	 ޷�A�*

logging/current_cost!G�;�P�+       ��K	�:޷�A�*

logging/current_costKG�;	�I�+       ��K	�m޷�A�*

logging/current_cost-G�;�8=�+       ��K	~�޷�A�*

logging/current_costPG�;@E�+       ��K	��޷�A�*

logging/current_costrG�;��Oe+       ��K	$�޷�A� *

logging/current_cost+G�;��+       ��K	�-߷�A� *

logging/current_costG�;)z+       ��K	b߷�A� *

logging/current_cost8G�;���?+       ��K	��߷�A� *

logging/current_cost$G�;�"�1+       ��K	��߷�A� *

logging/current_costG�;�4+�+       ��K	��A� *

logging/current_costG�;|��I+       ��K	R1��A� *

logging/current_costG�;�= +       ��K	t`��A� *

logging/current_cost G�;mZ6k+       ��K	z���A� *

logging/current_costG�;���g+       ��K	'���A� *

logging/current_costG�;3�GF+       ��K	����A� *

logging/current_cost1G�;���+       ��K	�0��A� *

logging/current_costG�;hd+       ��K	�_��A� *

logging/current_costG�;�>!+       ��K	Α��A� *

logging/current_costG�;��c�+       ��K	����A� *

logging/current_cost�F�;u'K+       ��K	����A� *

logging/current_cost�F�;
�+       ��K	�!��A� *

logging/current_costG�;�>�C+       ��K	/T��A� *

logging/current_cost	G�; +ؘ+       ��K	t���A� *

logging/current_cost�F�;�1��+       ��K	8���A� *

logging/current_cost�F�;8F$+       ��K	����A� *

logging/current_cost�F�;��x+       ��K	���A� *

logging/current_costG�;�۷+       ��K	�D��A� *

logging/current_cost�F�;͙:"+       ��K	]t��A� *

logging/current_cost�F�;�Tqw+       ��K	ڢ��A� *

logging/current_costG�;H��+       ��K	����A�!*

logging/current_cost�F�;���+       ��K	D���A�!*

logging/current_cost�F�;���l+       ��K	�,��A�!*

logging/current_cost�F�;��	+       ��K	�\��A�!*

logging/current_cost�F�;�Y9+       ��K	\���A�!*

logging/current_cost�F�;[���+       ��K	����A�!*

logging/current_cost�F�;����+       ��K	����A�!*

logging/current_cost�F�;�=Xu+       ��K	���A�!*

logging/current_cost�F�;'�+       ��K	�A��A�!*

logging/current_cost�F�;�� �+       ��K	6o��A�!*

logging/current_cost�F�;����+       ��K	����A�!*

logging/current_cost�F�;��8+       ��K	����A�!*

logging/current_cost�F�;7��6+       ��K	����A�!*

logging/current_cost�F�;�,-+       ��K	x'��A�!*

logging/current_cost�F�;<dQ�+       ��K	�V��A�!*

logging/current_cost�F�;+�@�+       ��K	����A�!*

logging/current_costG�;!r�+       ��K	0���A�!*

logging/current_cost-G�;3�ϥ+       ��K	t��A�!*

logging/current_costG�;2^+       ��K	�@��A�!*

logging/current_cost�F�;��;+       ��K	�n��A�!*

logging/current_cost�F�;����+       ��K	\���A�!*

logging/current_costG�;��3�+       ��K	����A�!*

logging/current_cost�F�;I�Z�+       ��K	b���A�!*

logging/current_cost�F�;w�@+       ��K	�0��A�!*

logging/current_cost�F�;QO�+       ��K	`��A�!*

logging/current_cost�F�;�� �+       ��K	����A�!*

logging/current_cost�F�;I!�0+       ��K	0���A�"*

logging/current_cost�F�;��;7+       ��K	����A�"*

logging/current_cost�F�;?���+       ��K	Y%��A�"*

logging/current_cost�F�;h� �+       ��K	*U��A�"*

logging/current_cost�F�;8���+       ��K	U���A�"*

logging/current_cost�F�;��+       ��K	����A�"*

logging/current_cost�F�;�Xg�+       ��K	���A�"*

logging/current_cost�F�;�B<�+       ��K	+-��A�"*

logging/current_cost�F�;���+       ��K	�b��A�"*

logging/current_cost�F�;⛯Z+       ��K	J���A�"*

logging/current_cost�F�;#�+       ��K	����A�"*

logging/current_cost~F�;�� �+       ��K	���A�"*

logging/current_cost|F�;^[��+       ��K	>��A�"*

logging/current_cost�F�;S0�+       ��K	nv��A�"*

logging/current_cost{F�;=��+       ��K	s���A�"*

logging/current_cost�F�;S�?+       ��K	^���A�"*

logging/current_cost�F�;�7�+       ��K	���A�"*

logging/current_cost�F�;��+       ��K		R��A�"*

logging/current_cost~F�;R��+       ��K	ӈ��A�"*

logging/current_cost�F�;����+       ��K	���A�"*

logging/current_cost�F�;�e��+       ��K	����A�"*

logging/current_costuF�;+�vg+       ��K	����A�"*

logging/current_costeF�;��,�+       ��K	eL���A�"*

logging/current_costuF�;y"�+       ��K	D����A�"*

logging/current_cost|F�;��+       ��K	Ư���A�"*

logging/current_cost�F�;��;�+       ��K	�����A�#*

logging/current_cost�F�;�v*�+       ��K	���A�#*

logging/current_cost[F�;��Q+       ��K	�:��A�#*

logging/current_cost�F�;I�*+       ��K	Yp��A�#*

logging/current_cost~F�; ��%+       ��K	���A�#*

logging/current_costZF�;�:;�+       ��K	����A�#*

logging/current_costSF�;�.��+       ��K		��A�#*

logging/current_costNF�;c溡+       ��K	Q6��A�#*

logging/current_costLF�;2�(n+       ��K	ug��A�#*

logging/current_costuF�;S�24+       ��K	ܙ��A�#*

logging/current_cost]F�;aUb2+       ��K	����A�#*

logging/current_costnF�;)�FP+       ��K	���A�#*

logging/current_costZF�;2�1�+       ��K	�*��A�#*

logging/current_cost^F�;�+       ��K	�Z��A�#*

logging/current_cost�F�;&US@+       ��K	I���A�#*

logging/current_costRF�;���+       ��K	����A�#*

logging/current_cost4F�;�W+       ��K	����A�#*

logging/current_costIF�;�~+       ��K	�*��A�#*

logging/current_costTF�;�Eɭ+       ��K	,\��A�#*

logging/current_cost;F�;d���+       ��K	Ǌ��A�#*

logging/current_costMF�;+��+       ��K	>���A�#*

logging/current_costdF�; Gr+       ��K	����A�#*

logging/current_costTF�;��ܡ+       ��K	4)��A�#*

logging/current_cost-F�;��@+       ��K	$]��A�#*

logging/current_costrF�;g�O�+       ��K	Η��A�#*

logging/current_costMF�;	[�+       ��K	!���A�#*

logging/current_cost�F�;�%�%+       ��K	 ��A�$*

logging/current_costoF�;:L�s+       ��K	3I��A�$*

logging/current_cost/F�;���%+       ��K	���A�$*

logging/current_costVF�;�\�c+       ��K	����A�$*

logging/current_costeF�;h|9d+       ��K	U���A�$*

logging/current_costUF�;�|+       ��K	q ���A�$*

logging/current_cost%F�;RB>�+       ��K	V���A�$*

logging/current_cost;F�;/���+       ��K	�����A�$*

logging/current_costF�;���i+       ��K	]����A�$*

logging/current_costF�;d��+       ��K	6����A�$*

logging/current_costF�;#�E3+       ��K	�)���A�$*

logging/current_costF�;e Ԍ+       ��K	�Y���A�$*

logging/current_cost%F�;�t�+       ��K	����A�$*

logging/current_costF�;��v+       ��K	�����A�$*

logging/current_costF�;��K�+       ��K	�����A�$*

logging/current_cost+F�;����+       ��K	g���A�$*

logging/current_costF�;�])~+       ��K	-C���A�$*

logging/current_costF�;P\G�+       ��K	�s���A�$*

logging/current_cost�E�;�gш+       ��K	~����A�$*

logging/current_costF�;� �v+       ��K	/����A�$*

logging/current_cost%F�;�tȋ+       ��K	�����A�$*

logging/current_cost�E�;�|��+       ��K	�-���A�$*

logging/current_costF�;��+       ��K	�]���A�$*

logging/current_costF�;����+       ��K	l����A�$*

logging/current_cost�E�;��Ǜ+       ��K	�����A�$*

logging/current_costF�;p�&�+       ��K	�����A�$*

logging/current_cost&F�;��9+       ��K	���A�%*

logging/current_costF�;iC��+       ��K	 D���A�%*

logging/current_cost�E�;�S�+       ��K	v���A�%*

logging/current_cost�E�;m��+       ��K	ϰ���A�%*

logging/current_cost�E�;�D׍+       ��K	�����A�%*

logging/current_costF�;�q+       ��K	H���A�%*

logging/current_costF�;Y�s�+       ��K	mN���A�%*

logging/current_cost�E�;X7��+       ��K	�����A�%*

logging/current_cost�E�;��.�+       ��K	-����A�%*

logging/current_cost�E�;�EX+       ��K	W����A�%*

logging/current_cost�E�;1ŀ�+       ��K	����A�%*

logging/current_cost�E�;6l�B+       ��K	NS���A�%*

logging/current_cost�E�;1��+       ��K	�����A�%*

logging/current_cost�E�;�U��+       ��K	�����A�%*

logging/current_cost�E�;�JX+       ��K	�����A�%*

logging/current_cost�E�;_���+       ��K	�1���A�%*

logging/current_costF�;�:�S+       ��K	h���A�%*

logging/current_cost�E�;E��`+       ��K	�?���A�%*

logging/current_costF�;6�5v+       ��K	�����A�%*

logging/current_cost5F�;��-�+       ��K	'����A�%*

logging/current_cost�E�;���W+       ��K	g���A�%*

logging/current_cost�E�;mr��+       ��K	�u���A�%*

logging/current_cost�E�;ȕ �+       ��K	k����A�%*

logging/current_cost�E�;ߞ�3+       ��K	�����A�%*

logging/current_cost�E�;yr6�+       ��K	�1���A�%*

logging/current_cost�E�;��sZ+       ��K	nn���A�&*

logging/current_cost�E�;�D*�+       ��K	����A�&*

logging/current_cost�E�;��!+       ��K	����A�&*

logging/current_cost�E�;\��+       ��K	����A�&*

logging/current_cost�E�;�c �+       ��K	'G���A�&*

logging/current_cost�E�;�'�N+       ��K	J{���A�&*

logging/current_cost�E�;
B�+       ��K	D����A�&*

logging/current_costF�;Q�� +       ��K	�����A�&*

logging/current_costF�;)�C+       ��K	� ��A�&*

logging/current_cost�E�;���+       ��K	�C ��A�&*

logging/current_cost�E�;a��6+       ��K	�q ��A�&*

logging/current_cost�E�;Q[��+       ��K	;� ��A�&*

logging/current_cost�E�;"̵�+       ��K	_� ��A�&*

logging/current_cost�E�;�
0>+       ��K	;��A�&*

logging/current_cost�E�;�-�+       ��K	O3��A�&*

logging/current_cost�E�;���+       ��K	ma��A�&*

logging/current_cost�E�;��Qt+       ��K	����A�&*

logging/current_cost�E�;��e�+       ��K	����A�&*

logging/current_cost�E�;�a�+       ��K	'���A�&*

logging/current_cost�E�;J+��+       ��K	g��A�&*

logging/current_cost�E�;��!�+       ��K	�L��A�&*

logging/current_cost�E�; \��+       ��K	�{��A�&*

logging/current_cost�E�;9
�+       ��K	?���A�&*

logging/current_cost�E�;�2�+       ��K	����A�&*

logging/current_cost�E�;"��@+       ��K	 ��A�&*

logging/current_cost�E�;�2�+       ��K	�C��A�&*

logging/current_costF�;�[�i+       ��K	�q��A�'*

logging/current_cost�E�;�σX+       ��K	Q���A�'*

logging/current_cost~E�;�N~+       ��K	{���A�'*

logging/current_cost�E�;�2��+       ��K	����A�'*

logging/current_cost�E�;�$_(+       ��K	r1��A�'*

logging/current_costsE�;6�ɰ+       ��K	Ib��A�'*

logging/current_cost�E�;·K�+       ��K	>���A�'*

logging/current_cost{E�;|� +       ��K	����A�'*

logging/current_cost|E�;�f|+       ��K	����A�'*

logging/current_cost�E�;q��+       ��K	���A�'*

logging/current_cost~E�;��G�+       ��K	&J��A�'*

logging/current_costuE�;,��g+       ��K	Hz��A�'*

logging/current_cost�E�;����+       ��K	m���A�'*

logging/current_costcE�;_��+       ��K	����A�'*

logging/current_cost�E�;�ò�+       ��K	M��A�'*

logging/current_cost�E�;�?i�+       ��K	`9��A�'*

logging/current_cost�E�;�9+       ��K	�j��A�'*

logging/current_cost�E�;|ka�+       ��K	ܙ��A�'*

logging/current_cost�E�;DU��+       ��K	����A�'*

logging/current_cost�E�;�r�+       ��K	���A�'*

logging/current_cost�E�;0��h+       ��K	�&��A�'*

logging/current_cost�E�;��L}+       ��K	�U��A�'*

logging/current_cost�E�;m���+       ��K	���A�'*

logging/current_cost�E�;�7zI+       ��K	|���A�'*

logging/current_costqE�;<ְS+       ��K	����A�'*

logging/current_cost�E�;�+       ��K	���A�(*

logging/current_cost�E�;\�G�+       ��K	�J��A�(*

logging/current_costzE�;���+       ��K	�{��A�(*

logging/current_cost|E�;r,��+       ��K	����A�(*

logging/current_cost�E�;ۿ��+       ��K	2���A�(*

logging/current_cost�E�;����+       ��K	8.	��A�(*

logging/current_cost|E�;c�/+       ��K	�e	��A�(*

logging/current_costzE�;�"B�+       ��K	d�	��A�(*

logging/current_cost�E�;u>�+       ��K	��	��A�(*

logging/current_cost�E�;����+       ��K	b�	��A�(*

logging/current_cost�E�;�}�|+       ��K	b+
��A�(*

logging/current_cost_E�;u�vm+       ��K	`
��A�(*

logging/current_costKE�;F�.++       ��K	U�
��A�(*

logging/current_costNE�;u��'+       ��K	y�
��A�(*

logging/current_costaE�;��|�+       ��K	�
��A�(*

logging/current_cost�E�;[��+       ��K	���A�(*

logging/current_cost�E�;]���+       ��K	oK��A�(*

logging/current_costaE�;2�~s+       ��K	�x��A�(*

logging/current_cost?E�;ƒ26+       ��K	ަ��A�(*

logging/current_costLE�;�l- +       ��K	����A�(*

logging/current_cost,E�;yQ�+       ��K	���A�(*

logging/current_costnE�;�_�+       ��K	�7��A�(*

logging/current_cost�E�;G�$�+       ��K	wh��A�(*

logging/current_costGE�;���+       ��K	_���A�(*

logging/current_costeE�;]�B+       ��K	&���A�(*

logging/current_costJE�;1�9�+       ��K	R���A�(*

logging/current_costE�;%� +       ��K	N!��A�)*

logging/current_costE�;���_+       ��K	TO��A�)*

logging/current_costME�;�{X�+       ��K	�|��A�)*

logging/current_costE�;35�?+       ��K	x���A�)*

logging/current_cost E�;'5#+       ��K	����A�)*

logging/current_costoE�;��Y�+       ��K	���A�)*

logging/current_costJE�;�5��+       ��K	17��A�)*

logging/current_costIE�;��+       ��K	�f��A�)*

logging/current_cost9E�;H�+       ��K	����A�)*

logging/current_costE�;s���+       ��K	���A�)*

logging/current_cost E�;%�)�+       ��K	����A�)*

logging/current_costlE�;�q9:+       ��K	� ��A�)*

logging/current_costmE�;̾\+       ��K	�K��A�)*

logging/current_cost�E�;cQ�L+       ��K	����A�)*

logging/current_cost]E�;WbK�+       ��K	T���A�)*

logging/current_costlE�;�'�+       ��K	��A�)*

logging/current_costrE�;c�+       ��K	'K��A�)*

logging/current_cost'E�;��+       ��K	ـ��A�)*

logging/current_cost}E�;�Qe�+       ��K	���A�)*

logging/current_cost0E�;Ω��+       ��K	I���A�)*

logging/current_costdE�;�n)+       ��K	�%��A�)*

logging/current_cost�E�;�o��+       ��K	�\��A�)*

logging/current_cost6E�;����+       ��K	����A�)*

logging/current_cost*E�;���+       ��K	o���A�)*

logging/current_costbE�;R�XL+       ��K	A���A�)*

logging/current_costiE�;U��&+       ��K	�0��A�)*

logging/current_costWE�;�<�+       ��K	 a��A�**

logging/current_cost3E�;g�+       ��K	����A�**

logging/current_costAE�;,+       ��K	o���A�**

logging/current_costE�;��:�+       ��K	[���A�**

logging/current_costE�;T�w�+       ��K	D#��A�**

logging/current_cost�D�;�+M`+       ��K	<T��A�**

logging/current_cost'E�;�0�=+       ��K	E���A�**

logging/current_cost6E�;鯃�+       ��K	����A�**

logging/current_costE�;�NZw+       ��K	����A�**

logging/current_costAE�;,L�+       ��K	�,��A�**

logging/current_cost�E�;�kz�+       ��K	�j��A�**

logging/current_costE�;2r�+       ��K	ښ��A�**

logging/current_costE�;�'��+       ��K	����A�**

logging/current_cost�D�;��J+       ��K	�&��A�**

logging/current_cost E�;���	+       ��K	�^��A�**

logging/current_cost�D�;��X�+       ��K	����A�**

logging/current_cost�D�;�%�+       ��K	*���A�**

logging/current_cost�D�;�}fg+       ��K	���A�**

logging/current_cost�D�;C϶�+       ��K	�\��A�**

logging/current_costE�;�D|�+       ��K	ծ��A�**

logging/current_cost�D�;�ܯ5+       ��K	s��A�**

logging/current_cost�D�;��I!+       ��K	�_��A�**

logging/current_cost�D�;U���+       ��K	|���A�**

logging/current_cost�D�;�E޶+       ��K	����A�**

logging/current_cost�D�;f.[�+       ��K	F��A�**

logging/current_costE�;@�+       ��K	�A��A�+*

logging/current_cost�D�;3N�C+       ��K	���A�+*

logging/current_cost�D�;D�A�+       ��K	p���A�+*

logging/current_cost�D�;���+       ��K	-���A�+*

logging/current_cost�D�;� )~+       ��K	�8��A�+*

logging/current_cost�D�;�S��+       ��K	�q��A�+*

logging/current_cost�D�;+{1(+       ��K	���A�+*

logging/current_cost�D�;�F�+       ��K	k���A�+*

logging/current_cost�D�;�(�+       ��K	$.��A�+*

logging/current_cost�D�;��\�+       ��K	b��A�+*

logging/current_cost�D�;$	Ӝ+       ��K	<���A�+*

logging/current_cost�D�;r�m+       ��K	1���A�+*

logging/current_cost�D�;�s#+       ��K	N��A�+*

logging/current_cost�D�;���%+       ��K	�Q��A�+*

logging/current_cost�D�;/Ѭ�+       ��K	����A�+*

logging/current_costE�;�T+       ��K	N���A�+*

logging/current_cost�D�;&�f+       ��K	s��A�+*

logging/current_cost�D�;���+       ��K	+b��A�+*

logging/current_costBE�;단2+       ��K	����A�+*

logging/current_cost�D�;�P�+       ��K	���A�+*

logging/current_costE�;�3�+       ��K	^D��A�+*

logging/current_cost0E�;���+       ��K	'���A�+*

logging/current_cost�D�;��5+       ��K	����A�+*

logging/current_cost�D�;=V�+       ��K	;���A�+*

logging/current_cost�D�;����+       ��K	5.��A�+*

logging/current_cost�D�;K>��+       ��K	�_��A�+*

logging/current_cost�D�;�Jߍ+       ��K	4���A�,*

logging/current_cost�D�;m��+       ��K	����A�,*

logging/current_costsD�; AC�+       ��K	� ��A�,*

logging/current_cost�D�;��s)+       ��K	E1��A�,*

logging/current_costwD�;����+       ��K	�i��A�,*

logging/current_cost�D�;I<+       ��K	ۼ��A�,*

logging/current_cost�D�;�:c�+       ��K	J���A�,*

logging/current_cost�D�;U1�B+       ��K	i! ��A�,*

logging/current_cost{D�;^41+       ��K	NT ��A�,*

logging/current_costXD�;Ͽ��+       ��K	� ��A�,*

logging/current_cost�D�;3��+       ��K	a� ��A�,*

logging/current_cost�D�;f<Ia+       ��K	�� ��A�,*

logging/current_costfD�;�<+       ��K	T/!��A�,*

logging/current_cost�D�;��+=+       ��K	�_!��A�,*

logging/current_cost�D�;�*�o+       ��K	Ď!��A�,*

logging/current_cost�D�;�*t�+       ��K	��!��A�,*

logging/current_cost�D�;rͽq+       ��K	��!��A�,*

logging/current_costLD�;
f�]+       ��K	F*"��A�,*

logging/current_cost�D�;f���+       ��K	l]"��A�,*

logging/current_cost�D�; Z1j+       ��K	~�"��A�,*

logging/current_cost�D�;��m�+       ��K	�"��A�,*

logging/current_cost�D�;��8+       ��K	D#��A�,*

logging/current_costQD�;�e
9+       ��K	�?#��A�,*

logging/current_cost^D�;+�+       ��K	�p#��A�,*

logging/current_costmD�;��d+       ��K	q�#��A�,*

logging/current_cost=D�;�/��+       ��K	��#��A�-*

logging/current_cost�D�;mW+       ��K	N$��A�-*

logging/current_cost%D�;f�+       ��K	M$��A�-*

logging/current_costuD�;6��+       ��K	\~$��A�-*

logging/current_cost�D�;M�9+       ��K	]�$��A�-*

logging/current_cost�D�;WqV�+       ��K	i�$��A�-*

logging/current_costLD�;'>�|+       ��K	�	%��A�-*

logging/current_cost�D�;�RK�+       ��K	�5%��A�-*

logging/current_costWD�;w(}�+       ��K	wd%��A�-*

logging/current_costD�;�3�+       ��K	)�%��A�-*

logging/current_costdD�;
'ip+       ��K	��%��A�-*

logging/current_costTD�;^���+       ��K	u�%��A�-*

logging/current_costND�;g<	�+       ��K	a&��A�-*

logging/current_cost�D�;D�+       ��K	�L&��A�-*

logging/current_cost4D�;U�Ml+       ��K	�y&��A�-*

logging/current_cost#D�;��R-+       ��K	A�&��A�-*

logging/current_costED�;wZ�+       ��K	��&��A�-*

logging/current_cost<D�;�m7+       ��K	_	'��A�-*

logging/current_cost]D�;�|fm+       ��K	�8'��A�-*

logging/current_costZD�;<��J+       ��K	e'��A�-*

logging/current_cost-D�;�S#1+       ��K	�'��A�-*

logging/current_cost�C�;��&+       ��K	1�'��A�-*

logging/current_cost*D�;��"+       ��K	q�'��A�-*

logging/current_costKD�;U�R7+       ��K	�(��A�-*

logging/current_costTD�;��P+       ��K	�L(��A�-*

logging/current_costND�;�)�+       ��K	�y(��A�-*

logging/current_costD�;W�d�+       ��K	M�(��A�.*

logging/current_cost4D�;�ڀ�+       ��K	|�(��A�.*

logging/current_cost>D�;���+       ��K	�)��A�.*

logging/current_costLD�;�z��+       ��K	t0)��A�.*

logging/current_costBD�;c%�+       ��K	#_)��A�.*

logging/current_cost�C�;��+       ��K	z�)��A�.*

logging/current_costD�;�bx�+       ��K	)�)��A�.*

logging/current_cost!D�;�Y��+       ��K	p�)��A�.*

logging/current_cost�C�;�E۩+       ��K	e*��A�.*

logging/current_costgD�;��-+       ��K	bB*��A�.*

logging/current_cost�C�;/|��+       ��K	Vp*��A�.*

logging/current_cost+D�;s���+       ��K	Z�*��A�.*

logging/current_costD�;,�t�+       ��K	s�*��A�.*

logging/current_cost�C�;L�5+       ��K	��*��A�.*

logging/current_costD�;s%��+       ��K	4+��A�.*

logging/current_cost-D�;&�l+       ��K	d+��A�.*

logging/current_costGD�;��ib+       ��K	�+��A�.*

logging/current_cost<D�;b�+       ��K	��+��A�.*

logging/current_cost�C�;O�.b+       ��K	��+��A�.*

logging/current_cost3D�;��j�+       ��K	�,��A�.*

logging/current_cost�C�;���+       ��K	�J,��A�.*

logging/current_cost�C�;� ڛ+       ��K	�y,��A�.*

logging/current_cost�C�;3���+       ��K	�,��A�.*

logging/current_costD�;e�+       ��K	 �,��A�.*

logging/current_cost�C�;��u�+       ��K	Y-��A�.*

logging/current_cost�C�;��~+       ��K	�9-��A�.*

logging/current_cost�C�;��w�+       ��K	�h-��A�/*

logging/current_costD�;��C+       ��K	ߗ-��A�/*

logging/current_costD�;�\��+       ��K	�-��A�/*

logging/current_cost�C�;=? �+       ��K	��-��A�/*

logging/current_cost�C�;p�^+       ��K	<#.��A�/*

logging/current_cost�C�;�o$I+       ��K	�R.��A�/*

logging/current_cost�C�;jb+       ��K	��.��A�/*

logging/current_cost�C�;]��+       ��K	P�.��A�/*

logging/current_cost�C�;"X�+       ��K	��.��A�/*

logging/current_cost�C�;c�Zb+       ��K	�/��A�/*

logging/current_cost�C�;'h�+       ��K	�=/��A�/*

logging/current_cost�C�;�9|K+       ��K	Xl/��A�/*

logging/current_cost�C�;q�'�+       ��K	��/��A�/*

logging/current_cost�C�;�K2.+       ��K	�/��A�/*

logging/current_costC�;��V+       ��K	k�/��A�/*

logging/current_cost�C�;��{�+       ��K	�&0��A�/*

logging/current_cost�C�;̬Y�+       ��K	CZ0��A�/*

logging/current_cost�C�;��+       ��K	+�0��A�/*

logging/current_cost�C�;��<~+       ��K	��0��A�/*

logging/current_cost�C�;:h�H+       ��K	��0��A�/*

logging/current_cost�C�;�k�T+       ��K	�1��A�/*

logging/current_cost�C�;�Ek-+       ��K	?J1��A�/*

logging/current_cost�C�;}fʟ+       ��K	~x1��A�/*

logging/current_cost�C�;5�2s+       ��K	ʦ1��A�/*

logging/current_cost�C�;��|=+       ��K	T�1��A�/*

logging/current_cost�C�;���s+       ��K	s2��A�0*

logging/current_costYC�;��+       ��K	�22��A�0*

logging/current_cost�C�;
pd+       ��K	�a2��A�0*

logging/current_cost�C�;e]�+       ��K	D�2��A�0*

logging/current_costaC�;�F�+       ��K	M�2��A�0*

logging/current_cost{C�;��!+       ��K	R�2��A�0*

logging/current_cost�C�;I?��+       ��K	V3��A�0*

logging/current_cost�C�;����+       ��K	H3��A�0*

logging/current_costrC�;�u��+       ��K	�u3��A�0*

logging/current_costQC�;̣��+       ��K	��3��A�0*

logging/current_costrC�;���+       ��K	��3��A�0*

logging/current_cost�C�;v-��+       ��K	'4��A�0*

logging/current_costmC�;�oy�+       ��K	t54��A�0*

logging/current_cost�C�;�Cڻ+       ��K	�f4��A�0*

logging/current_costSC�;"_�+       ��K	�4��A�0*

logging/current_cost_C�;թ��+       ��K	��4��A�0*

logging/current_costHC�;���+       ��K	��4��A�0*

logging/current_cost7C�;fg;E+       ��K	�5��A�0*

logging/current_cost1C�;��+       ��K	�R5��A�0*

logging/current_costyC�;;x�#+       ��K	��5��A�0*

logging/current_cost�C�;s��+       ��K	��5��A�0*

logging/current_cost/C�;AŜ�+       ��K	T�5��A�0*

logging/current_costJC�;ɻ��+       ��K	6��A�0*

logging/current_costPC�;9�G+       ��K	�V6��A�0*

logging/current_cost�C�;�_�+       ��K	��6��A�0*

logging/current_costD�;Ά[+       ��K	�6��A�0*

logging/current_cost�C�;�v�+       ��K	�*7��A�1*

logging/current_cost�C�;~��+       ��K	Od7��A�1*

logging/current_cost�C�;p���+       ��K	G�7��A�1*

logging/current_costUC�;�;�q+       ��K	��7��A�1*

logging/current_cost�C�;L�*+       ��K	9�7��A�1*

logging/current_cost�C�;U@"&+       ��K	L08��A�1*

logging/current_cost�C�;|P+       ��K	�x8��A�1*

logging/current_costC�;��+       ��K	��8��A�1*

logging/current_costC�;�/+       ��K	��8��A�1*

logging/current_cost.C�;�nKW+       ��K	�(9��A�1*

logging/current_cost"C�;��nQ+       ��K	:\9��A�1*

logging/current_costC�;k��+       ��K	G�9��A�1*

logging/current_costpC�;�K0�+       ��K	��9��A�1*

logging/current_costQC�;��;�+       ��K	$:��A�1*

logging/current_cost�B�;?$ZT+       ��K	�5:��A�1*

logging/current_costC�;(fh7+       ��K	)d:��A�1*

logging/current_cost�B�;3��+       ��K	�:��A�1*

logging/current_costPC�;p�Tj+       ��K	��:��A�1*

logging/current_costeC�;�y-+       ��K	d�:��A�1*

logging/current_cost�C�;t�{�+       ��K	k#;��A�1*

logging/current_cost#D�;���+       ��K	�T;��A�1*

logging/current_cost�C�;<�l�+       ��K	��;��A�1*

logging/current_cost�B�;����+       ��K	U<��A�1*

logging/current_cost�B�;vŋ�+       ��K	�9<��A�1*

logging/current_cost�B�;-�b3+       ��K	l<��A�1*

logging/current_cost�B�;L�+       ��K	ҟ<��A�2*

logging/current_costuC�;��-+       ��K	*�<��A�2*

logging/current_cost�B�;j<�+       ��K	�=��A�2*

logging/current_cost�B�;vē+       ��K	�9=��A�2*

logging/current_cost�B�;��=C+       ��K	.h=��A�2*

logging/current_cost�B�;.���+       ��K	U�=��A�2*

logging/current_cost�B�;C귾+       ��K	��=��A�2*

logging/current_costC�;�^��+       ��K	��=��A�2*

logging/current_costTC�;6�{+       ��K	0>��A�2*

logging/current_cost�B�;��W+       ��K	�b>��A�2*

logging/current_cost�B�;�J5�+       ��K	��>��A�2*

logging/current_cost�B�;E�V�+       ��K	<�>��A�2*

logging/current_cost�B�;N�+       ��K	�
?��A�2*

logging/current_costC�;j*�+       ��K	�G?��A�2*

logging/current_cost�B�;S�dp+       ��K	7y?��A�2*

logging/current_cost�B�;2a��+       ��K	&�?��A�2*

logging/current_costC�;}��+       ��K	��?��A�2*

logging/current_cost�B�;�+       ��K	�@��A�2*

logging/current_cost�B�;򈘡+       ��K	�4@��A�2*

logging/current_cost�B�;&���+       ��K	�`@��A�2*

logging/current_cost�B�;�d��+       ��K	��@��A�2*

logging/current_cost�B�;���+       ��K	�@��A�2*

logging/current_cost�B�;|��L+       ��K	��@��A�2*

logging/current_cost�B�;ٖ/_+       ��K	gA��A�2*

logging/current_cost�B�;Y$��+       ��K	jEA��A�2*

logging/current_cost�B�;A:}�+       ��K	tA��A�2*

logging/current_cost�B�;�u7�+       ��K	ϡA��A�3*

logging/current_cost�B�;�+��+       ��K	��A��A�3*

logging/current_cost�B�;S%�g+       ��K	��A��A�3*

logging/current_cost�B�;��+       ��K	*B��A�3*

logging/current_cost�B�;��+       ��K	-XB��A�3*

logging/current_costjB�;�;�+       ��K	�B��A�3*

logging/current_cost_B�;���+       ��K	2�B��A�3*

logging/current_cost�B�;CT��+       ��K	�B��A�3*

logging/current_costSB�;����+       ��K	�C��A�3*

logging/current_costvB�;=�<�+       ��K	�KC��A�3*

logging/current_cost�B�;Ok�4+       ��K	�{C��A�3*

logging/current_costQB�;m��+       ��K	�C��A�3*

logging/current_cost�B�;��=+       ��K	��C��A�3*

logging/current_costoB�;ȋ�f+       ��K	�D��A�3*

logging/current_costMB�;/�#+       ��K	�0D��A�3*

logging/current_cost6B�;�}4�+       ��K	�^D��A�3*

logging/current_cost�B�;��+       ��K	'�D��A�3*

logging/current_costB�;%�R+       ��K	ϻD��A�3*

logging/current_cost5B�;OL�+       ��K	:�D��A�3*

logging/current_costOB�;Z��
+       ��K	4E��A�3*

logging/current_cost'B�;��M�+       ��K	�KE��A�3*

logging/current_costrB�;0Cb�+       ��K	�~E��A�3*

logging/current_cost-B�;)l+       ��K	,�E��A�3*

logging/current_costYB�;"8�u+       ��K	��E��A�3*

logging/current_cost^B�;�tZ�+       ��K	�F��A�3*

logging/current_cost�B�;���+       ��K	�:F��A�3*

logging/current_cost(B�;�N�+       ��K	5hF��A�4*

logging/current_costIB�;��aQ+       ��K	��F��A�4*

logging/current_costsB�;��:+       ��K	��F��A�4*

logging/current_cost�B�;Y�:�+       ��K	s�F��A�4*

logging/current_costfB�;*>�^+       ��K	!G��A�4*

logging/current_cost0B�;��Id+       ��K	�OG��A�4*

logging/current_costB�;˕݆+       ��K	{}G��A�4*

logging/current_cost5B�;��+       ��K	g�G��A�4*

logging/current_cost�A�;)/�s+       ��K	��G��A�4*

logging/current_costB�;s\�+       ��K	\H��A�4*

logging/current_costdB�;�@��+       ��K	8H��A�4*

logging/current_costC�;ElfT+       ��K	GkH��A�4*

logging/current_cost{B�;{љc+       ��K	��H��A�4*

logging/current_cost�B�;��W%+       ��K	]�H��A�4*

logging/current_cost�B�;g�+       ��K	dI��A�4*

logging/current_costEB�;��D�+       ��K	IOI��A�4*

logging/current_costnB�;}6J�+       ��K	�I��A�4*

logging/current_costB�;Pjl�+       ��K	T�I��A�4*

logging/current_costB�;�L<�+       ��K	2J��A�4*

logging/current_cost�A�;e/��+       ��K	LLJ��A�4*

logging/current_cost�A�;Ė��+       ��K	�J��A�4*

logging/current_cost�A�;�"}+       ��K	�J��A�4*

logging/current_cost�A�;|N�+       ��K	]K��A�4*

logging/current_cost�B�;bE��+       ��K	NAK��A�4*

logging/current_cost�A�;zB!+       ��K	�wK��A�4*

logging/current_costFB�;�#L�+       ��K	]�K��A�5*

logging/current_cost�B�;DAg+       ��K	/�K��A�5*

logging/current_costaB�;O0�^+       ��K	�L��A�5*

logging/current_costvB�;��SY+       ��K	@SL��A�5*

logging/current_cost�B�;���9+       ��K	��L��A�5*

logging/current_cost�A�;�G:�+       ��K	$�L��A�5*

logging/current_costB�;�i+       ��K	��L��A�5*

logging/current_costEB�;�ֻ+       ��K	� M��A�5*

logging/current_costjB�;CU(
+       ��K	�NM��A�5*

logging/current_costyA�;�E�V+       ��K	|~M��A�5*

logging/current_cost�A�;b.#+       ��K	}�M��A�5*

logging/current_cost�A�;��+       ��K	��M��A�5*

logging/current_cost�A�;U��+       ��K	�N��A�5*

logging/current_cost�A�;)�n�+       ��K	Y=N��A�5*

logging/current_cost�A�;�d"%+       ��K	�mN��A�5*

logging/current_cost
B�;���+       ��K	j�N��A�5*

logging/current_costEA�;�9�+       ��K	��N��A�5*

logging/current_cost�A�;>J�}+       ��K	�N��A�5*

logging/current_cost B�;�O�+       ��K	U(O��A�5*

logging/current_cost�A�;'�++       ��K	UUO��A�5*

logging/current_costB�;^(�+       ��K	։O��A�5*

logging/current_cost?A�;�+ܔ+       ��K	�O��A�5*

logging/current_cost!A�;�Zl+       ��K	��O��A�5*

logging/current_cost�A�;�Ƞy+       ��K	�P��A�5*

logging/current_costUA�;�a�+       ��K	mLP��A�5*

logging/current_costmA�;)~�#+       ��K	OzP��A�5*

logging/current_cost
A�;����+       ��K	-�P��A�6*

logging/current_cost�A�;b��B+       ��K	��P��A�6*

logging/current_cost�@�;:���+       ��K	�Q��A�6*

logging/current_cost\A�;�dc�+       ��K	�>Q��A�6*

logging/current_cost�A�;'�n!+       ��K	WlQ��A�6*

logging/current_cost�A�;Ҳ�k+       ��K	]�Q��A�6*

logging/current_cost-A�;_u!N+       ��K	��Q��A�6*

logging/current_cost�@�;�s+       ��K	*�Q��A�6*

logging/current_cost@A�;w�)�+       ��K	�'R��A�6*

logging/current_costA�;�(.+       ��K	�YR��A�6*

logging/current_cost�A�;D�&c+       ��K	��R��A�6*

logging/current_cost�A�;��)+       ��K	,�R��A�6*

logging/current_cost�A�;g��p+       ��K	!�R��A�6*

logging/current_costgA�;�|a�+       ��K	�&S��A�6*

logging/current_cost�@�;��QY+       ��K	US��A�6*

logging/current_costrA�;�m�+       ��K	�S��A�6*

logging/current_cost�@�;�Ԏ+       ��K	&�S��A�6*

logging/current_cost�@�;I�x+       ��K	��S��A�6*

logging/current_costJA�;��+       ��K	y!T��A�6*

logging/current_cost�@�;�7�+       ��K	�UT��A�6*

logging/current_cost�@�;�>f+       ��K	ڈT��A�6*

logging/current_cost�@�;vo0�+       ��K	f�T��A�6*

logging/current_cost�@�;���+       ��K	sU��A�6*

logging/current_cost�@�;��+       ��K	�;U��A�6*

logging/current_cost�@�;�g~v+       ��K	&rU��A�6*

logging/current_cost�@�;=G�z+       ��K	_�U��A�7*

logging/current_cost�@�;>���+       ��K	��U��A�7*

logging/current_cost�@�;���+       ��K	�V��A�7*

logging/current_costjA�;�\U+       ��K	7RV��A�7*

logging/current_cost�@�;_+2+       ��K	�V��A�7*

logging/current_cost�@�;^+�C+       ��K	Y�V��A�7*

logging/current_costA�;/��+       ��K	.�V��A�7*

logging/current_cost�@�;f|+       ��K	pW��A�7*

logging/current_cost�@�;�#=�+       ��K	)MW��A�7*

logging/current_costh@�;8�N+       ��K	�|W��A�7*

logging/current_cost�A�;L)�n+       ��K	ѮW��A�7*

logging/current_costk@�;�1��+       ��K	��W��A�7*

logging/current_cost�@�;y54+       ��K	�X��A�7*

logging/current_costA�;�VW�+       ��K	UBX��A�7*

logging/current_costq@�;/��+       ��K	qsX��A�7*

logging/current_cost;A�;"g�+       ��K	��X��A�7*

logging/current_cost�@�;?��,+       ��K	��X��A�7*

logging/current_cost�A�;��+       ��K	<Y��A�7*

logging/current_cost�A�;b�+       ��K	86Y��A�7*

logging/current_costFA�;蠤!+       ��K	�iY��A�7*

logging/current_cost�A�;y�r�+       ��K	��Y��A�7*

logging/current_costU@�;��|+       ��K	��Y��A�7*

logging/current_cost�@�;�t�+       ��K	H�Y��A�7*

logging/current_costj@�;�S~�+       ��K	�,Z��A�7*

logging/current_costD@�;�_X+       ��K	3^Z��A�7*

logging/current_cost,@�;�i�5+       ��K	T�Z��A�7*

logging/current_cost!@�;��+       ��K	��Z��A�8*

logging/current_cost3@�;=�+       ��K	��Z��A�8*

logging/current_cost�?�;C��+       ��K	~"[��A�8*

logging/current_cost @�;�I�+       ��K	�~[��A�8*

logging/current_costp@�;�=��+       ��K	@�[��A�8*

logging/current_cost�@�;Q�U�+       ��K	��[��A�8*

logging/current_cost�@�;�]�+       ��K	 "\��A�8*

logging/current_cost@�;踋+       ��K	KX\��A�8*

logging/current_cost�?�;co&R+       ��K	j�\��A�8*

logging/current_cost5@�;ĳ��+       ��K	��\��A�8*

logging/current_cost�?�;5�=�+       ��K	�]��A�8*

logging/current_cost @�;�1��+       ��K	 ?]��A�8*

logging/current_cost�?�;p�5+       ��K	�t]��A�8*

logging/current_cost$@�;1E�$+       ��K	կ]��A�8*

logging/current_costl@�;#��+       ��K	�]��A�8*

logging/current_cost�@�;�D�+       ��K	�#^��A�8*

logging/current_cost�@�;�<+       ��K	�^^��A�8*

logging/current_cost�?�;���+       ��K	�^��A�8*

logging/current_costC@�;�e�+       ��K	&�^��A�8*

logging/current_cost�@�;H/`�+       ��K	�_��A�8*

logging/current_costb@�;��˧+       ��K	IC_��A�8*

logging/current_cost�@�;��>�+       ��K	A~_��A�8*

logging/current_cost�?�;�Hہ+       ��K	��_��A�8*

logging/current_cost�?�;'6��+       ��K	��_��A�8*

logging/current_cost�?�;��à+       ��K	� `��A�8*

logging/current_cost�?�;X�<�+       ��K	?]`��A�8*

logging/current_cost2@�;����+       ��K	��`��A�9*

logging/current_cost�?�;�9T�+       ��K	k�`��A�9*

logging/current_cost�?�;JlϚ+       ��K	�	a��A�9*

logging/current_cost|@�;]
J+       ��K	E@a��A�9*

logging/current_cost�?�;X�/+       ��K	ua��A�9*

logging/current_cost�?�;�2g+       ��K	��a��A�9*

logging/current_cost�?�;����+       ��K	��a��A�9*

logging/current_cost�?�;Ik��+       ��K	�b��A�9*

logging/current_cost�?�;�rr+       ��K	iBb��A�9*

logging/current_cost�?�;4�X�+       ��K	6ob��A�9*

logging/current_costw?�;ka"+       ��K	��b��A�9*

logging/current_cost�?�;M��+       ��K	�b��A�9*

logging/current_cost>@�;h�̳+       ��K	c��A�9*

logging/current_costr?�;�8�+       ��K	�0c��A�9*

logging/current_cost\?�;��7+       ��K	g`c��A�9*

logging/current_cost?�;G�!�+       ��K	c�c��A�9*

logging/current_cost@�;"Ԃ�+       ��K	&�c��A�9*

logging/current_costa@�;���+       ��K	��c��A�9*

logging/current_cost�@�;CAs_+       ��K	�d��A�9*

logging/current_cost�?�;e���+       ��K	]Ld��A�9*

logging/current_cost?�;����+       ��K	C{d��A�9*

logging/current_cost^?�;C�
+       ��K	^�d��A�9*

logging/current_cost?�;Q2F+       ��K	�d��A�9*

logging/current_cost�?�;�Y:+       ��K	�e��A�9*

logging/current_cost?�;�ά+       ��K	\:e��A�9*

logging/current_costZ?�;�hr�+       ��K	�ie��A�:*

logging/current_costy?�;@/r�+       ��K	k�e��A�:*

logging/current_cost�?�;娤u+       ��K	q�e��A�:*

logging/current_cost�>�;�4+       ��K	�e��A�:*

logging/current_cost�>�;f��+       ��K	#&f��A�:*

logging/current_costG?�;��8x+       ��K	�Vf��A�:*

logging/current_costm?�;�`;�+       ��K	��f��A�:*

logging/current_cost�?�;I��+       ��K	�f��A�:*

logging/current_cost�>�;���+       ��K	E�f��A�:*

logging/current_cost�>�;;a�+       ��K	tg��A�:*

logging/current_cost?�;�x�+       ��K	dBg��A�:*

logging/current_cost�>�;��e+       ��K	�qg��A�:*

logging/current_cost ?�;�O��+       ��K	ɟg��A�:*

logging/current_cost�>�;�T��+       ��K	��g��A�:*

logging/current_cost�>�;R1�g+       ��K	��g��A�:*

logging/current_cost�>�;F�{c+       ��K	�,h��A�:*

logging/current_cost?�;�!j�+       ��K	1Zh��A�:*

logging/current_cost�>�;�B11+       ��K	��h��A�:*

logging/current_cost�>�;_���+       ��K	�h��A�:*

logging/current_cost�>�;����+       ��K	J�h��A�:*

logging/current_cost�>�;�R-�+       ��K	�i��A�:*

logging/current_cost�>�;��vc+       ��K	9>i��A�:*

logging/current_cost6>�;�<��+       ��K	)mi��A�:*

logging/current_costB>�;��h�+       ��K	�i��A�:*

logging/current_cost�>�;�O$+       ��K	+�i��A�:*

logging/current_cost�>�;��H}+       ��K	 �i��A�:*

logging/current_costU>�;��,�+       ��K	�&j��A�;*

logging/current_cost�>�;��S�+       ��K	9Sj��A�;*

logging/current_cost�>�;����+       ��K	��j��A�;*

logging/current_cost/?�;�+�+       ��K	�j��A�;*

logging/current_cost�>�;F��+       ��K	a�j��A�;*

logging/current_cost >�;�[�+       ��K	?	k��A�;*

logging/current_cost�>�;c�v+       ��K	6k��A�;*

logging/current_cost�=�;�7�+       ��K	)ck��A�;*

logging/current_cost!?�;^�d +       ��K	đk��A�;*

logging/current_cost@>�;9�Ǥ+       ��K	P�k��A�;*

logging/current_cost�=�;Cv��+       ��K	��k��A�;*

logging/current_cost>�;�4_+       ��K	Dl��A�;*

logging/current_cost/>�;c��+       ��K	8Ll��A�;*

logging/current_cost�=�;���+       ��K	�zl��A�;*

logging/current_cost�=�;#��+       ��K	��l��A�;*

logging/current_cost�=�;�>��+       ��K	��l��A�;*

logging/current_cost�=�;i�1�+       ��K	�
m��A�;*

logging/current_cost�=�;d�+       ��K	k9m��A�;*

logging/current_cost�=�;�H͈+       ��K	�gm��A�;*

logging/current_cost�=�;TƏ+       ��K	�m��A�;*

logging/current_cost>�;���+       ��K	��m��A�;*

logging/current_cost�>�;f�+       ��K	��m��A�;*

logging/current_costY?�;�a�*+       ��K	"n��A�;*

logging/current_cost�>�;[+       ��K	7Tn��A�;*

logging/current_cost?�;����+       ��K	΁n��A�;*

logging/current_cost.?�;Z��+       ��K	�n��A�<*

logging/current_cost>�;��j+       ��K	��n��A�<*

logging/current_costz>�;�Ǻ5+       ��K	�o��A�<*

logging/current_costF>�;Wh�=+       ��K	�9o��A�<*

logging/current_cost�>�;��-+       ��K	�fo��A�<*

logging/current_cost=�;�!��+       ��K	M�o��A�<*

logging/current_cost�=�;1�+       ��K	v�o��A�<*

logging/current_cost�=�;F���+       ��K	O�o��A�<*

logging/current_costo=�;���+       ��K	�&p��A�<*

logging/current_cost|=�;��sZ+       ��K	Wp��A�<*

logging/current_cost�=�;��h�+       ��K	%�p��A�<*

logging/current_cost�=�;�%�~+       ��K	d�p��A�<*

logging/current_cost=�;�*Ø+       ��K	��p��A�<*

logging/current_cost=�;!)�+       ��K	�q��A�<*

logging/current_cost�=�;yN�+       ��K	�Iq��A�<*

logging/current_cost�=�;���+       ��K	�wq��A�<*

logging/current_cost<>�;�B��+       ��K	ӧq��A�<*

logging/current_costX=�;�$�c+       ��K	��q��A�<*

logging/current_cost5=�;Ƚ~�+       ��K	rr��A�<*

logging/current_cost�=�;���J+       ��K	�<r��A�<*

logging/current_cost&=�;t\g~+       ��K	�ir��A�<*

logging/current_costP=�;��a�+       ��K	�r��A�<*

logging/current_cost =�;MO�R+       ��K	�r��A�<*

logging/current_costM=�;/>�W+       ��K	��r��A�<*

logging/current_cost�=�;u�B�+       ��K	�.s��A�<*

logging/current_costM=�;\�C�+       ��K	�\s��A�<*

logging/current_cost=�;�1J�+       ��K	T�s��A�=*

logging/current_cost=�;&�ش+       ��K	�s��A�=*

logging/current_cost�<�;�W
!+       ��K	��s��A�=*

logging/current_cost|<�;��+       ��K	V(t��A�=*

logging/current_costv<�;�JU$+       ��K	�\t��A�=*

logging/current_cost�<�;��3p+       ��K	��t��A�=*

logging/current_cost�<�;݋B+       ��K	��t��A�=*

logging/current_cost=�;^'T+       ��K	��t��A�=*

logging/current_cost�<�;�H��+       ��K	�"u��A�=*

logging/current_cost�<�;�1X�+       ��K	#Ru��A�=*

logging/current_costI<�;��q+       ��K	>�u��A�=*

logging/current_cost�<�;��+       ��K	��u��A�=*

logging/current_costJ<�;����+       ��K	�u��A�=*

logging/current_cost�<�;C?п+       ��K	Bv��A�=*

logging/current_cost+<�;����+       ��K	4Bv��A�=*

logging/current_cost?<�;���+       ��K	.rv��A�=*

logging/current_cost<�;��+       ��K	E�v��A�=*

logging/current_cost�<�;�G�+       ��K	��v��A�=*

logging/current_cost)=�;}�+       ��K	��v��A�=*

logging/current_cost�=�;�X �+       ��K	U+w��A�=*

logging/current_cost�;�;�V��+       ��K	�Xw��A�=*

logging/current_cost�<�;j���+       ��K	��w��A�=*

logging/current_cost�;�;,S+       ��K	�w��A�=*

logging/current_cost�<�;<-��+       ��K	|�w��A�=*

logging/current_cost=<�;@�ť+       ��K	�x��A�=*

logging/current_cost�<�;���+       ��K	1Bx��A�=*

logging/current_cost/<�;<�c+       ��K	ipx��A�>*

logging/current_costy<�;�w�+       ��K	Ϟx��A�>*

logging/current_cost�=�;j؉+       ��K	��x��A�>*

logging/current_cost�=�;u	;+       ��K	��x��A�>*

logging/current_cost?=�;��+       ��K	�+y��A�>*

logging/current_costd<�;e�6�+       ��K	�Zy��A�>*

logging/current_cost�<�;̸��+       ��K	B�y��A�>*

logging/current_cost�;�;�	a+       ��K	�y��A�>*

logging/current_cost�;�;$�:�+       ��K	-�y��A�>*

logging/current_costx;�;.���+       ��K	~z��A�>*

logging/current_cost_<�;fw��+       ��K	�Pz��A�>*

logging/current_costV<�;�*'�+       ��K	�}z��A�>*

logging/current_cost�<�;
�JI+       ��K	V�z��A�>*

logging/current_cost�;�;����+       ��K	1�z��A�>*

logging/current_cost|;�;Q;+       ��K	�
{��A�>*

logging/current_cost�;�;�㻴+       ��K	�;{��A�>*

logging/current_cost_;�;�!�5+       ��K	�k{��A�>*

logging/current_costr;�;�Av+       ��K	��{��A�>*

logging/current_cost�;�;MR�d+       ��K	�5|��A�>*

logging/current_cost�;�;a�ڝ+       ��K	N�|��A�>*

logging/current_cost�:�;Ñ1+       ��K	��|��A�>*

logging/current_cost�:�;HS�+       ��K	�}��A�>*

logging/current_cost3;�;o"j+       ��K	�B}��A�>*

logging/current_costz;�;�᪔+       ��K	��}��A�>*

logging/current_cost	;�;W��2+       ��K	��}��A�>*

logging/current_cost6;�;�$�M+       ��K	��}��A�?*

logging/current_cost�;�;�� �+       ��K	�.~��A�?*

logging/current_cost0<�;5�t�+       ��K	�e~��A�?*

logging/current_cost�;�;j���+       ��K	�~��A�?*

logging/current_cost�:�;�E�+       ��K	 �~��A�?*

logging/current_cost�;�;��+       ��K	���A�?*

logging/current_cost�<�;d�l�+       ��K	�6��A�?*

logging/current_costI<�;uV��+       ��K	Fo��A�?*

logging/current_costE<�;_5�+       ��K	r���A�?*

logging/current_cost<�;F�,+       ��K	����A�?*

logging/current_cost+;�;n"��+       ��K	����A�?*

logging/current_costt;�;�9��+       ��K	�>���A�?*

logging/current_costg;�;>�v+       ��K	�o���A�?*

logging/current_cost�:�;U�
+       ��K	~����A�?*

logging/current_cost�:�;<�RK+       ��K	�π��A�?*

logging/current_cost�:�;����+       ��K	y���A�?*

logging/current_cost�:�;)섹+       ��K	0���A�?*

logging/current_cost^:�;��!#+       ��K	�]���A�?*

logging/current_costf:�;�Ee�+       ��K	.����A�?*

logging/current_cost�9�;��+       ��K	�����A�?*

logging/current_costN:�;8!+       ��K	3쁸�A�?*

logging/current_costS;�;Zlv�+       ��K	u���A�?*

logging/current_cost�9�;1=9n+       ��K	\X���A�?*

logging/current_cost�9�;�.�i+       ��K	�����A�?*

logging/current_cost:�;i��+       ��K	{Ђ��A�?*

logging/current_cost�9�;).Q+       ��K	����A�?*

logging/current_cost�:�;1q%+       ��K	V���A�@*

logging/current_cost;�;?<�$+       ��K	�����A�@*

logging/current_cost4;�;p��+       ��K	̃��A�@*

logging/current_cost:�;���+       ��K	h����A�@*

logging/current_cost�9�;1��+       ��K	�3���A�@*

logging/current_cost�9�;��9�+       ��K	�n���A�@*

logging/current_cost�9�;��Z+       ��K	�����A�@*

logging/current_cost�9�;:ow+       ��K	k儸�A�@*

logging/current_cost�:�;�m+       ��K	����A�@*

logging/current_cost:�;ڛZ�+       ��K	SL���A�@*

logging/current_costE;�;	I=+       ��K	�����A�@*

logging/current_cost;�;�8|9+       ��K	�����A�@*

logging/current_cost�:�;R��+       ��K	����A�@*

logging/current_cost29�;�H+       ��K	�-���A�@*

logging/current_costm9�;,o�+       ��K	�\���A�@*

logging/current_cost�8�;خ�J+       ��K	o����A�@*

logging/current_cost�9�;@�+       ��K	%І��A�@*

logging/current_costf:�;���+       ��K	����A�@*

logging/current_cost7:�;t��[+       ��K	8���A�@*

logging/current_cost�8�;���+       ��K	�g���A�@*

logging/current_cost�8�;{� =+       ��K	砇��A�@*

logging/current_costt9�;CN�;+       ��K	Vч��A�@*

logging/current_cost69�;�>��+       ��K	����A�@*

logging/current_costb9�;sY�+       ��K	BC���A�@*

logging/current_cost�8�;'��~+       ��K	�|���A�@*

logging/current_cost_9�;r�۽+       ��K	ָ���A�A*

logging/current_cost�8�;MŐ+       ��K	�爸�A�A*

logging/current_cost{8�;�p��+       ��K	X���A�A*

logging/current_cost�9�;���+       ��K	)O���A�A*

logging/current_cost�:�;��+       ��K	Z����A�A*

logging/current_cost�:�;Ϫ�+       ��K	 É��A�A*

logging/current_cost�9�;�W�+       ��K	��A�A*

logging/current_cost9�;��c7+       ��K	%���A�A*

logging/current_cost�9�;���!+       ��K	�]���A�A*

logging/current_cost�8�;�!k�+       ��K	�����A�A*

logging/current_costd8�;���+       ��K	�ۊ��A�A*

logging/current_cost9�;�ud+       ��K	E���A�A*

logging/current_cost.8�;���+       ��K	�?���A�A*

logging/current_cost]8�;�Do�+       ��K	q���A�A*

logging/current_cost8�;��c�+       ��K	c����A�A*

logging/current_cost8�;d	0I+       ��K	Ӌ��A�A*

logging/current_cost&9�;Zs��+       ��K	q���A�A*

logging/current_costl8�;�+       ��K	+4���A�A*

logging/current_cost�7�;"��*+       ��K	i���A�A*

logging/current_cost8�;��+       ��K	�����A�A*

logging/current_cost�7�;d�y+       ��K	�ˌ��A�A*

logging/current_cost�7�;���+       ��K	����A�A*

logging/current_cost�8�;~��+       ��K	�(���A�A*

logging/current_cost�7�;����+       ��K	"V���A�A*

logging/current_cost�7�;����+       ��K	�����A�A*

logging/current_cost�7�;,�X�+       ��K	;����A�A*

logging/current_cost�7�;_W<+       ��K	A荸�A�B*

logging/current_costa7�;E��+       ��K	����A�B*

logging/current_cost�7�;s}��+       ��K	`H���A�B*

logging/current_cost�6�;�+��+       ��K	by���A�B*

logging/current_cost7�;��9r+       ��K	����A�B*

logging/current_costE8�;��$x+       ��K	`َ��A�B*

logging/current_cost�7�;�B~k+       ��K	����A�B*

logging/current_costb7�;A�+       ��K	�3���A�B*

logging/current_costx7�;zJO+       ��K	Yc���A�B*

logging/current_cost�6�;ls}�+       ��K	�����A�B*

logging/current_cost~7�;����+       ��K	я��A�B*

logging/current_cost%7�;N�c�+       ��K	D����A�B*

logging/current_cost�6�;�@+       ��K	�-���A�B*

logging/current_cost�7�;v�N+       ��K	7`���A�B*

logging/current_cost�6�;L(�k+       ��K	\����A�B*

logging/current_cost�7�;��R�+       ��K	�̐��A�B*

logging/current_cost�8�;����+       ��K	`���A�B*

logging/current_cost�8�;U���+       ��K	�0���A�B*

logging/current_cost�8�;���+       ��K	�_���A�B*

logging/current_cost7�;:� +       ��K	�����A�B*

logging/current_cost�7�;ɼ��+       ��K	�����A�B*

logging/current_cost?7�;3��+       ��K	c푸�A�B*

logging/current_cost�6�;"�+       ��K	����A�B*

logging/current_cost�6�;���+       ��K	�I���A�B*

logging/current_cost�6�;���+       ��K	x���A�B*

logging/current_cost�7�;\E��+       ��K	ʧ���A�B*

logging/current_cost�7�;�K�+       ��K	�ْ��A�C*

logging/current_costn6�;��H+       ��K	�	���A�C*

logging/current_cost�5�;�!�}+       ��K	9���A�C*

logging/current_cost16�;�$I}+       ��K	:m���A�C*

logging/current_costv6�;JW�
+       ��K	ʟ���A�C*

logging/current_cost�5�;X`��+       ��K	<Г��A�C*

logging/current_cost:6�;�+       ��K	j���A�C*

logging/current_cost�5�;�r�+       ��K	�5���A�C*

logging/current_cost�5�;u��+       ��K	Md���A�C*

logging/current_cost�6�;��Ȅ+       ��K	����A�C*

logging/current_cost�5�;���a+       ��K	�����A�C*

logging/current_cost�6�;ߞ��+       ��K	����A�C*

logging/current_cost�7�;Ϗ��+       ��K	�'���A�C*

logging/current_cost�7�;�6��+       ��K	NX���A�C*

logging/current_cost7�;��Vu+       ��K	{����A�C*

logging/current_cost5�;��+       ��K	ổ��A�C*

logging/current_cost-5�;��+       ��K	�핸�A�C*

logging/current_cost�4�;�g+       ��K	����A�C*

logging/current_cost�4�;u���+       ��K	�L���A�C*

logging/current_cost_5�;���+       ��K	x|���A�C*

logging/current_cost5�;�_�~+       ��K	穖��A�C*

logging/current_costf5�;���+       ��K	�ז��A�C*

logging/current_cost�4�;�2Rv+       ��K	U���A�C*

logging/current_cost�4�;]��?+       ��K	C���A�C*

logging/current_cost6�;�u�+       ��K	rp���A�C*

logging/current_cost5�;5o�+       ��K	 ����A�D*

logging/current_cost_4�;�X^�+       ��K	nҗ��A�D*

logging/current_cost�4�;�?�k+       ��K	���A�D*

logging/current_cost4�;�cÖ+       ��K	=6���A�D*

logging/current_cost4�;��^�+       ��K	e���A�D*

logging/current_cost�3�;�7�;+       ��K	�����A�D*

logging/current_cost4�;#��+       ��K	�Ș��A�D*

logging/current_cost�5�;�ȯ+       ��K	�����A�D*

logging/current_cost 4�;JF��+       ��K	�#���A�D*

logging/current_cost�3�;x�SG+       ��K	HY���A�D*

logging/current_cost�3�;��	�+       ��K	É���A�D*

logging/current_costq3�;�K�+       ��K	n����A�D*

logging/current_costI4�;�>��+       ��K	a虸�A�D*

logging/current_cost4�;��|+       ��K	����A�D*

logging/current_cost�3�;͎WL+       ��K	�G���A�D*

logging/current_costz3�;�q�+       ��K	lu���A�D*

logging/current_costq3�;��ރ+       ��K	�����A�D*

logging/current_cost!3�;�_O~+       ��K	�֚��A�D*

logging/current_costT4�;-�e�+       ��K	����A�D*

logging/current_cost�4�;4j�w+       ��K	J2���A�D*

logging/current_cost�4�;(�y�+       ��K	�`���A�D*

logging/current_costb3�;�4��+       ��K	U����A�D*

logging/current_cost)3�;�n�e+       ��K	����A�D*

logging/current_cost�3�;�':+       ��K	c꛸�A�D*

logging/current_costN3�;�L}�+       ��K	����A�D*

logging/current_cost�2�;�0�+       ��K	G���A�D*

logging/current_cost3�;���m+       ��K	s���A�E*

logging/current_cost�3�;t+       ��K	�����A�E*

logging/current_cost�3�;q�Z+       ��K	'Μ��A�E*

logging/current_costi2�;�(�	+       ��K	����A�E*

logging/current_cost�3�;Tx��+       ��K	m1���A�E*

logging/current_cost�2�;�L�+       ��K	�_���A�E*

logging/current_cost(2�;M���+       ��K	0����A�E*

logging/current_cost`2�;���+       ��K	�����A�E*

logging/current_costM2�;Zn�+       ��K	�ꝸ�A�E*

logging/current_cost�1�;�.(Y+       ��K	����A�E*

logging/current_cost�3�;���+       ��K	�F���A�E*

logging/current_cost2�;j}��+       ��K	7t���A�E*

logging/current_cost�2�;�"4+       ��K	$����A�E*

logging/current_cost�1�; �7'+       ��K	�̞��A�E*

logging/current_cost+3�;|v��+       ��K	z����A�E*

logging/current_cost�2�;%���+       ��K	))���A�E*

logging/current_cost�1�;��@+       ��K	AV���A�E*

logging/current_cost12�;*1�+       ��K	Є���A�E*

logging/current_costU1�;�	=�+       ��K	p����A�E*

logging/current_cost�0�;Y�J+       ��K	Cߟ��A�E*

logging/current_cost�1�;�تW+       ��K	I���A�E*

logging/current_costf1�;���H+       ��K	6<���A�E*

logging/current_costi1�;O��+       ��K	�m���A�E*

logging/current_costn2�;��X�+       ��K	�����A�E*

logging/current_cost�3�;ۊ&�+       ��K	mР��A�E*

logging/current_cost3�;3a?}+       ��K	N����A�F*

logging/current_cost�1�;��k+       ��K	�+���A�F*

logging/current_costQ2�;��g.+       ��K	;Y���A�F*

logging/current_cost1�;.���+       ��K	�����A�F*

logging/current_cost�0�;�ya++       ��K	�����A�F*

logging/current_costs0�;�+       ��K	�塸�A�F*

logging/current_cost�0�;�]�D+       ��K	����A�F*

logging/current_cost�0�;�=Q+       ��K	�B���A�F*

logging/current_cost�0�;v��#+       ��K	�n���A�F*

logging/current_costB1�;c���+       ��K	`����A�F*

logging/current_cost1�;`�+       ��K	�Ϣ��A�F*

logging/current_cost/0�;X28Q+       ��K	�����A�F*

logging/current_costn0�;��P+       ��K	�*���A�F*

logging/current_cost'2�;��8+       ��K	SY���A�F*

logging/current_cost�2�;����+       ��K	����A�F*

logging/current_cost�2�;~ɮ7+       ��K	����A�F*

logging/current_costl0�;	/w+       ��K	�壸�A�F*

logging/current_cost�0�;�3`B+       ��K	����A�F*

logging/current_costP/�;p��+       ��K	�B���A�F*

logging/current_costu1�;j�[+       ��K	Pr���A�F*

logging/current_cost�/�;z9�1+       ��K	c����A�F*

logging/current_cost/�;H��&+       ��K	6ͤ��A�F*

logging/current_cost[0�;�O�o+       ��K	w����A�F*

logging/current_cost�.�;��� +       ��K	c*���A�F*

logging/current_coste0�;u���+       ��K	�X���A�F*

logging/current_cost�0�;aY�P+       ��K	����A�F*

logging/current_costO1�;	��+       ��K	�����A�G*

logging/current_coste2�;U�d^+       ��K	�祸�A�G*

logging/current_cost�0�;���+       ��K	����A�G*

logging/current_cost/�;f��+       ��K	�I���A�G*

logging/current_cost�0�;jҖ]+       ��K	n{���A�G*

logging/current_costN/�;��+       ��K	����A�G*

logging/current_cost�/�;��+       ��K	�Ԧ��A�G*

logging/current_costT.�;h{+       ��K	4
���A�G*

logging/current_costm.�;GE�6+       ��K	>9���A�G*

logging/current_cost�.�;+�+       ��K	~e���A�G*

logging/current_cost.�;v�Y}+       ��K	'����A�G*

logging/current_cost.�;s1�r+       ��K	Z§��A�G*

logging/current_cost.�;��3+       ��K	���A�G*

logging/current_cost�-�;�+�+       ��K	�!���A�G*

logging/current_costz-�;5ي+       ��K	<Q���A�G*

logging/current_cost-�;A�}x+       ��K	����A�G*

logging/current_cost�-�;�,y+       ��K	�����A�G*

logging/current_cost�-�;7�,�+       ��K	kܨ��A�G*

logging/current_cost�-�;ڹ#�+       ��K	
���A�G*

logging/current_cost�-�;t�MI+       ��K	�8���A�G*

logging/current_cost�-�;c��+       ��K	Gj���A�G*

logging/current_cost�.�;'�m�+       ��K	=����A�G*

logging/current_cost0�;g��~+       ��K	�ũ��A�G*

logging/current_costw0�;�纆+       ��K	���A�G*

logging/current_cost00�;��%+       ��K	����A�G*

logging/current_cost0�;JZ\`+       ��K	|N���A�G*

logging/current_costX-�;.A�6+       ��K	Hz���A�H*

logging/current_cost�-�;%A��+       ��K	V����A�H*

logging/current_cost�-�;Z�C�+       ��K	eԪ��A�H*

logging/current_costC.�;w/Q}+       ��K	����A�H*

logging/current_cost,�;��I+       ��K	�3���A�H*

logging/current_cost�,�;���+       ��K		b���A�H*

logging/current_cost�.�;��ͪ+       ��K	����A�H*

logging/current_cost/�;;���+       ��K	3����A�H*

logging/current_cost2,�;��7V+       ��K	�����A�H*

logging/current_cost�+�;��$�+       ��K	O���A�H*

logging/current_cost�,�;o8W�+       ��K	PI���A�H*

logging/current_cost�,�;=��+       ��K	 w���A�H*

logging/current_cost�*�;����+       ��K	?����A�H*

logging/current_cost-�;�Î�+       ��K	�Ԭ��A�H*

logging/current_costb-�;��+       ��K	����A�H*

logging/current_cost`.�;�"�a+       ��K	�0���A�H*

logging/current_cost�*�;���+       ��K	r]���A�H*

logging/current_cost�,�;V[m�+       ��K	2����A�H*

logging/current_cost�*�;��f�+       ��K	�����A�H*

logging/current_cost�*�;�N�+       ��K	�����A�H*

logging/current_cost.+�;�E%�+       ��K	{���A�H*

logging/current_cost�+�;�3x�+       ��K	zJ���A�H*

logging/current_cost�*�;�k+       ��K	�y���A�H*

logging/current_cost$,�;�uH~+       ��K	�����A�H*

logging/current_cost�,�;�z5�+       ��K	�Ԯ��A�H*

logging/current_cost,�;��K�+       ��K	 ���A�I*

logging/current_cost`+�;�ȁ�+       ��K	5���A�I*

logging/current_costm*�;���X+       ��K	/c���A�I*

logging/current_cost�*�;-�+       ��K	z����A�I*

logging/current_cost�+�;�wv)+       ��K	¯��A�I*

logging/current_cost�)�;Us+       ��K	���A�I*

logging/current_cost�)�;sk�+       ��K	�"���A�I*

logging/current_cost�)�;��E�+       ��K	�Q���A�I*

logging/current_cost�(�;�O�+       ��K	q����A�I*

logging/current_cost�(�;���/+       ��K	]����A�I*

logging/current_cost�)�;���+       ��K	#۰��A�I*

logging/current_cost�(�;��?+       ��K	����A�I*

logging/current_cost)�;�-��+       ��K	�C���A�I*

logging/current_cost)�;^�Bd+       ��K	4t���A�I*

logging/current_cost�)�;�:�i+       ��K	V����A�I*

logging/current_costY(�;���+       ��K	�ױ��A�I*

logging/current_cost�(�;y��+       ��K	�
���A�I*

logging/current_cost*�;-,�{+       ��K	+?���A�I*

logging/current_cost|(�;�Ǎ+       ��K	�n���A�I*

logging/current_cost�)�;����+       ��K	�����A�I*

logging/current_costQ*�;6J>�+       ��K	�ز��A�I*

logging/current_cost�'�;>#��+       ��K	,	���A�I*

logging/current_cost�(�;b��+       ��K	�:���A�I*

logging/current_costv(�;S/*�+       ��K	7q���A�I*

logging/current_costl)�;���E+       ��K	6����A�I*

logging/current_cost�'�;��\�+       ��K	<ҳ��A�I*

logging/current_cost9)�;�=�+       ��K	(���A�J*

logging/current_cost�'�;�q��+       ��K	(5���A�J*

logging/current_cost?)�;�۳�+       ��K	Oi���A�J*

logging/current_cost�)�;��+       ��K	�����A�J*

logging/current_cost�(�;�N�T+       ��K	2Ǵ��A�J*

logging/current_cost�'�;�7��+       ��K	�����A�J*

logging/current_cost#'�;�/%+       ��K	�&���A�J*

logging/current_costG&�;w� +       ��K	>V���A�J*

logging/current_cost	'�;�_�+       ��K	�����A�J*

logging/current_cost�&�;m���+       ��K	����A�J*

logging/current_cost}'�;�c��+       ��K	�䵸�A�J*

logging/current_cost�&�;��jC+       ��K	3���A�J*

logging/current_cost�&�;���+       ��K	�@���A�J*

logging/current_cost~&�;�j+       ��K	o���A�J*

logging/current_costi&�;C]�r+       ��K	/����A�J*

logging/current_costQ'�;|�~+       ��K	�ʶ��A�J*

logging/current_cost�)�;�?�l+       ��K	8����A�J*

logging/current_costW(�;�b�+       ��K	<&���A�J*

logging/current_cost1(�;*��+       ��K	DS���A�J*

logging/current_cost�%�;�PW+       ��K	]����A�J*

logging/current_cost�$�;�
k+       ��K	߮���A�J*

logging/current_cost�$�;m�CT+       ��K	Rܷ��A�J*

logging/current_cost�%�;�=I�+       ��K	
���A�J*

logging/current_cost�$�;z��+       ��K	z8���A�J*

logging/current_cost�$�;�e�+       ��K	f���A�J*

logging/current_cost�$�;\|��+       ��K	U����A�K*

logging/current_cost�%�;� �z+       ��K	�����A�K*

logging/current_cost:'�;�i��+       ��K	��A�K*

logging/current_cost~'�;��x&+       ��K	����A�K*

logging/current_cost�$�;��P�+       ��K	BJ���A�K*

logging/current_costV$�;��4e+       ��K	Iy���A�K*

logging/current_cost�#�;����+       ��K	����A�K*

logging/current_cost9#�;���l+       ��K	dӹ��A�K*

logging/current_cost�#�;@�
E+       ��K	� ���A�K*

logging/current_cost�$�;3.�+       ��K	�-���A�K*

logging/current_cost�"�;�tf�+       ��K	�]���A�K*

logging/current_cost$#�;�
�+       ��K	c����A�K*

logging/current_cost?$�;A%I+       ��K	`˺��A�K*

logging/current_cost�"�;�/�+       ��K	����A�K*

logging/current_cost�"�;�h�+       ��K	>B���A�K*

logging/current_cost7#�;���@+       ��K	�	���A�K*

logging/current_cost�#�;���y+       ��K	
Z���A�K*

logging/current_cost�"�;���+       ��K	�ټ��A�K*

logging/current_cost�!�;�(+       ��K	*���A�K*

logging/current_costx"�;>��8+       ��K	YX���A�K*

logging/current_cost�"�;:<U+       ��K	�O���A�K*

logging/current_costv$�;gG��+       ��K	�¾��A�K*

logging/current_cost�!�;]��+       ��K	N���A�K*

logging/current_costL!�;H���+       ��K	�E���A�K*

logging/current_cost�"�; ���+       ��K	R{���A�K*

logging/current_cost)!�;_hc6+       ��K	2����A�K*

logging/current_costT!�;0gpk+       ��K	�ݿ��A�L*

logging/current_costi �;(g	J+       ��K	r���A�L*

logging/current_costh �;F�+       ��K	L���A�L*

logging/current_cost�!�;�1;+       ��K	u����A�L*

logging/current_costU!�;/~�+       ��K	>����A�L*

logging/current_costj!�; u��+       ��K	�����A�L*

logging/current_cost� �;� +       ��K	<���A�L*

logging/current_cost� �;����+       ��K	�V���A�L*

logging/current_cost� �;2,��+       ��K	\����A�L*

logging/current_cost; �;�Ū�+       ��K	#����A�L*

logging/current_cost��;i��+       ��K	�¸�A�L*

logging/current_cost1 �;�j%+       ��K	�9¸�A�L*

logging/current_cost��;�	I�+       ��K	�l¸�A�L*

logging/current_cost��;O?k�+       ��K	e�¸�A�L*

logging/current_cost��;��+       ��K	+�¸�A�L*

logging/current_cost� �;�"�+       ��K	�	ø�A�L*

logging/current_costR �;J{��+       ��K	h>ø�A�L*

logging/current_cost��;,&� +       ��K	3nø�A�L*

logging/current_cost!�;�%��+       ��K	��ø�A�L*

logging/current_cost��;�f�+       ��K	�ø�A�L*

logging/current_cost��;T9b�+       ��K	��ø�A�L*

logging/current_cost_ �;�:~�+       ��K	0ĸ�A�L*

logging/current_costv�;���l+       ��K	�bĸ�A�L*

logging/current_cost��;*��(+       ��K	�ĸ�A�L*

logging/current_cost�;�D��+       ��K	j�ĸ�A�L*

logging/current_costk�;�'��+       ��K	W�ĸ�A�L*

logging/current_cost�;�Gd�+       ��K	�3Ÿ�A�M*

logging/current_cost�;��ʆ+       ��K	�hŸ�A�M*

logging/current_cost/�;'�g�+       ��K	�Ÿ�A�M*

logging/current_cost^�;�!�+       ��K	��Ÿ�A�M*

logging/current_cost��;*�Y+       ��K	�Ÿ�A�M*

logging/current_costh�;�(�b+       ��K	�'Ƹ�A�M*

logging/current_costc�;��+       ��K	�UƸ�A�M*

logging/current_cost��;���+       ��K	 �Ƹ�A�M*

logging/current_cost��;��LL+       ��K	�Ƹ�A�M*

logging/current_costp�;���U+       ��K	T�Ƹ�A�M*

logging/current_cost�;z��+       ��K	'Ǹ�A�M*

logging/current_costq�;9�u+       ��K	�JǸ�A�M*

logging/current_cost��;�Vo/+       ��K	xǸ�A�M*

logging/current_costJ�;��Y;+       ��K	��Ǹ�A�M*

logging/current_cost��;\���+       ��K	�Ǹ�A�M*

logging/current_cost��;���L+       ��K	�ȸ�A�M*

logging/current_cost��;}M	�+       ��K	�<ȸ�A�M*

logging/current_cost��;��N+       ��K	�sȸ�A�M*

logging/current_cost��;tl3!+       ��K	�ȸ�A�M*

logging/current_cost�;��^�+       ��K	��ȸ�A�M*

logging/current_cost��;�{�;+       ��K	�ɸ�A�M*

logging/current_cost��;��@B+       ��K	�8ɸ�A�M*

logging/current_cost��;(�+       ��K	\pɸ�A�M*

logging/current_cost�;�r4�+       ��K	۠ɸ�A�M*

logging/current_costw�;�v��+       ��K	�ɸ�A�M*

logging/current_cost��;�.�+       ��K	_ ʸ�A�N*

logging/current_cost�;���+       ��K	�/ʸ�A�N*

logging/current_cost;�;���+       ��K	�oʸ�A�N*

logging/current_cost��;
ƕ�