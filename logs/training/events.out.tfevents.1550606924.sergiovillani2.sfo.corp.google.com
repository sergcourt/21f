       �K"	   ��Abrain.Event:2�~gM�      ��	��)��A"��
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
seed2 *
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_1/weights1
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
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
z
layer_1/biases1/readIdentitylayer_1/biases1*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
�
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*'
_output_shapes
:���������*
T0
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
seed2 *
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_2/weights2
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
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2
�
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
layer_2/weights2/readIdentitylayer_2/weights2*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
!layer_2/biases2/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_2/biases2*
valueB*    
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
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
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
!layer_3/biases3/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
layer_3/biases3
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
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
z
layer_3/biases3/readIdentitylayer_3/biases3*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
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
seed2 *
dtype0*
_output_shapes

:*

seed *
T0*"
_class
loc:@output/weights4
�
.output/weights4/Initializer/random_uniform/subSub.output/weights4/Initializer/random_uniform/max.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes
: 
�
.output/weights4/Initializer/random_uniform/mulMul8output/weights4/Initializer/random_uniform/RandomUniform.output/weights4/Initializer/random_uniform/sub*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
output/weights4
VariableV2*"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
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
output/biases4/readIdentityoutput/biases4*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
output/MatMulMatMullayer_3/Reluoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
g

output/addAddoutput/MatMuloutput/biases4/read*'
_output_shapes
:���������*
T0
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
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/cost/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
z
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
_output_shapes
:*
T0*
out_type0
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
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*'
_output_shapes
:���������*
T0
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
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
�
/train/gradients/cost/SquaredDifference_grad/mulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*'
_output_shapes
:���������*
T0
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
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg
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
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
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
&train/gradients/layer_3/add_grad/Sum_1Sum*train/gradients/layer_3/Relu_grad/ReluGrad8train/gradients/layer_3/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
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
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1
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
train/beta1_power/readIdentitytrain/beta1_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *"
_class
loc:@layer_1/biases1*
valueB
 *w�?
�
train/beta2_power
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
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
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
#train/layer_1/biases1/Adam_1/AssignAssigntrain/layer_1/biases1/Adam_1.train/layer_1/biases1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
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
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2
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
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container 
�
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
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
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
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
!train/layer_3/biases3/Adam/AssignAssigntrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
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
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container 
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
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
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
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
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
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
use_nesterov( *
_output_shapes
:
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
+train/Adam/update_layer_2/biases2/ApplyAdam	ApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_2/biases2*
use_nesterov( *
_output_shapes
:
�
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@layer_3/weights3
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
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@output/weights4*
use_nesterov( *
_output_shapes

:
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
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking( 
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
logging/current_costScalarSummarylogging/current_cost/tags	cost/Mean*
_output_shapes
: *
T0
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
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*(
dtypes
2*|
_output_shapesj
h::::::::::::::::::::::::::
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
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
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
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
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
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
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
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
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
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign"��v��     ��d]	�s.��AJ܉
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
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*'
_output_shapes
:���������*
T0
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
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
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
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
z
layer_2/biases2/readIdentitylayer_2/biases2*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
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
1layer_3/weights3/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@layer_3/weights3*
valueB"      
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
seed2 *
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_3/weights3
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
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/cost/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
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
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
p
&train/gradients/cost/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
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
#train/gradients/cost/Mean_grad/CastCast'train/gradients/cost/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
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
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
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
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
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
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg
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
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
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
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
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
train/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *"
_class
loc:@layer_1/biases1*
valueB
 *w�?
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
-train/layer_1/weights1/Adam/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam
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
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
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
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
"train/layer_1/weights1/Adam_1/readIdentitytrain/layer_1/weights1/Adam_1*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
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
 train/layer_2/weights2/Adam/readIdentitytrain/layer_2/weights2/Adam*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
/train/layer_2/weights2/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_2/weights2*
valueB*    
�
train/layer_2/weights2/Adam_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2
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
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
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
#train/layer_2/biases2/Adam_1/AssignAssigntrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
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
"train/layer_3/weights3/Adam/AssignAssigntrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
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
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
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
.train/output/weights4/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*"
_class
loc:@output/weights4*
valueB*    
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
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
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
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@layer_1/weights1*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
use_nesterov( *
_output_shapes
:
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
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@layer_3/weights3
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
*train/Adam/update_output/biases4/ApplyAdam	ApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*!
_class
loc:@output/biases4
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
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1
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
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
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
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
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
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�
save/Assign_10Assigntrain/layer_1/biases1/Adamsave/RestoreV2:10*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
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
save/Assign_13Assigntrain/layer_1/weights1/Adam_1save/RestoreV2:13*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
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
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
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
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
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
train/output/biases4/Adam_1:0"train/output/biases4/Adam_1/Assign"train/output/biases4/Adam_1/read:02/train/output/biases4/Adam_1/Initializer/zeros:0d�z(       �pJ	2�2��A*

logging/current_costz	�=���*       ����	33��A*

logging/current_costl��=�m�<*       ����	�b3��A
*

logging/current_cost�`�=ڵ�*       ����	>�3��A*

logging/current_cost�0�=\\��*       ����	��3��A*

logging/current_cost�˝=7G�*       ����	��3��A*

logging/current_costxؐ=�(V�*       ����	�"4��A*

logging/current_cost���= L�*       ����	�P4��A#*

logging/current_costA}x=��*       ����	#�4��A(*

logging/current_cost�(h=���7*       ����	�4��A-*

logging/current_cost�xY=O��*       ����	��4��A2*

logging/current_cost\L=���g*       ����	5��A7*

logging/current_cost�5A=Wka*       ����	�85��A<*

logging/current_costl6=7P��*       ����	�g5��AA*

logging/current_cost��,=gy*       ����	m�5��AF*

logging/current_cost��$=�ܭ*       ����	�5��AK*

logging/current_costp=��`X*       ����	,�5��AP*

logging/current_costU=�%��*       ����	76��AU*

logging/current_cost�1=��*       ����	�J6��AZ*

logging/current_cost�=��I)*       ����	�y6��A_*

logging/current_cost�}
=B�*       ����	�6��Ad*

logging/current_cost��=&�L�*       ����	��6��Ai*

logging/current_cost�=)\�
*       ����	�7��An*

logging/current_cost]r =M��A*       ����	�-7��As*

logging/current_cost�7�<CT�e*       ����	Z7��Ax*

logging/current_cost���<덦�*       ����	��7��A}*

logging/current_cost_*�<v�p�+       ��K	��7��A�*

logging/current_cost���<A�ӄ+       ��K	
�7��A�*

logging/current_costD��<���+       ��K	*8��A�*

logging/current_cost�/�<Âș+       ��K	�?8��A�*

logging/current_cost@�<�)u+       ��K	hm8��A�*

logging/current_cost.*�<��O�+       ��K	��8��A�*

logging/current_cost���<�>��+       ��K	��8��A�*

logging/current_cost��<���+       ��K	��8��A�*

logging/current_cost ��<
�v�+       ��K	�/9��A�*

logging/current_cost=��<v���+       ��K	�a9��A�*

logging/current_costm��<pO
�+       ��K	�9��A�*

logging/current_cost4c�<��:�+       ��K	0�9��A�*

logging/current_costbS�<]��+       ��K	t�9��A�*

logging/current_cost��<����+       ��K	�&:��A�*

logging/current_costE4�<�(?�+       ��K	HZ:��A�*

logging/current_cost�`�<o���+       ��K	��:��A�*

logging/current_cost���<c�g�+       ��K	�:��A�*

logging/current_cost֮�<� �^+       ��K	i�:��A�*

logging/current_cost�՘<�@��+       ��K	Q;��A�*

logging/current_cost�ٓ<��f�+       ��K	J;��A�*

logging/current_cost�ʎ<�O��+       ��K	��;��A�*

logging/current_costD��<�kD+       ��K	�;��A�*

logging/current_cost���<�E��+       ��K	�<��A�*

logging/current_cost�<Q��6+       ��K	�W<��A�*

logging/current_cost��t<�%;+       ��K	@�<��A�*

logging/current_cost��j<���4+       ��K	��<��A�*

logging/current_cost��`<��� +       ��K	i=��A�*

logging/current_costQ�V<�X�Y+       ��K	W=��A�*

logging/current_cost|uL<:Ӻ�+       ��K	��=��A�*

logging/current_cost
�B<�5I+       ��K	,�=��A�*

logging/current_costj�8<�A�+       ��K	�>��A�*

logging/current_cost�Q/<��͋+       ��K	�N>��A�*

logging/current_cost '&<"��3+       ��K	��>��A�*

logging/current_cost�M<\/�K+       ��K	��>��A�*

logging/current_cost��<: a�+       ��K	��>��A�*

logging/current_cost��<��_�+       ��K	�?��A�*

logging/current_cost�R<Eu�+       ��K	�k?��A�*

logging/current_cost���;���;+       ��K	?�?��A�*

logging/current_cost�R�;���}+       ��K	��?��A�*

logging/current_costI��;y�(�+       ��K	>@��A�*

logging/current_costr�;+R�+       ��K	\;@��A�*

logging/current_cost���;o}3�+       ��K	�p@��A�*

logging/current_cost���;�̭+       ��K	ǩ@��A�*

logging/current_cost��;�H��+       ��K	��@��A�*

logging/current_cost�7�;�4�+       ��K	5A��A�*

logging/current_cost��;��W+       ��K	�>A��A�*

logging/current_cost8'�;�6$B+       ��K	KpA��A�*

logging/current_cost��;8��+       ��K	��A��A�*

logging/current_cost��;�u8T+       ��K	�A��A�*

logging/current_cost��;��2�+       ��K	�B��A�*

logging/current_cost���;~�\+       ��K	*SB��A�*

logging/current_cost��;3��+       ��K	��B��A�*

logging/current_cost0��;�u�L+       ��K	$�B��A�*

logging/current_cost:r�;j�#�+       ��K	$�B��A�*

logging/current_cost��;�.��+       ��K	�.C��A�*

logging/current_cost*�;Ź|+       ��K	EaC��A�*

logging/current_costg}�;(��+       ��K	B�C��A�*

logging/current_cost�4�;����+       ��K	�C��A�*

logging/current_cost0�;@�+       ��K	��C��A�*

logging/current_cost9�;&{_+       ��K	�)D��A�*

logging/current_cost"'�;���+       ��K	�dD��A�*

logging/current_cost�W�;��4+       ��K	_�D��A�*

logging/current_cost���;C�~+       ��K	��D��A�*

logging/current_cost���;_�'+       ��K	� E��A�*

logging/current_cost�j�;݊�{+       ��K	1E��A�*

logging/current_cost^��;��3�+       ��K	�cE��A�*

logging/current_cost���;0sUL+       ��K	ƖE��A�*

logging/current_costU2�;��v+       ��K	�E��A�*

logging/current_cost�ݒ;�q��+       ��K	DF��A�*

logging/current_cost���;���L+       ��K	4F��A�*

logging/current_costmI�;/[M�+       ��K		gF��A�*

logging/current_cost��;�?��+       ��K	��F��A�*

logging/current_costʑ;%tς+       ��K	�F��A�*

logging/current_cost���;���+       ��K	;G��A�*

logging/current_cost=Z�;p�er+       ��K	߅G��A�*

logging/current_cost�%�;�`��+       ��K	B�G��A�*

logging/current_cost��;��s+       ��K	�	H��A�*

logging/current_cost�Đ;��)E+       ��K	.LH��A�*

logging/current_cost��;���
+       ��K	E�H��A�*

logging/current_cost�j�;�Q��+       ��K	�H��A�*

logging/current_cost%6�;����+       ��K	[I��A�*

logging/current_cost���;��G+       ��K	�DI��A�*

logging/current_cost�ˏ;���`+       ��K	��I��A�*

logging/current_cost��;�Gк+       ��K	öI��A�*

logging/current_costQ}�;�;�+       ��K	^�I��A�*

logging/current_cost�Y�;�o�+       ��K	@!J��A�*

logging/current_cost�7�;+x�+       ��K	VJ��A�*

logging/current_cost �;�rѣ+       ��K	k�J��A�*

logging/current_cost��;�v��+       ��K	I�J��A�*

logging/current_cost.��;�B$N+       ��K	f�J��A�*

logging/current_cost��;��K+       ��K	�#K��A�*

logging/current_cost;V�;�ƞ�+       ��K	qYK��A�*

logging/current_cost'�;DE��+       ��K	4�K��A�*

logging/current_cost���;���@+       ��K	��K��A�*

logging/current_cost:Ս;:;�+       ��K	��K��A�*

logging/current_cost`��;�eT+       ��K	�)L��A�*

logging/current_cost���;1�"+       ��K	�ZL��A�*

logging/current_cost��;��/�+       ��K	$�L��A�*

logging/current_costSh�;F;?Q+       ��K	��L��A�*

logging/current_cost�R�;[���+       ��K	F�L��A�*

logging/current_cost`?�;��+       ��K	�:M��A�*

logging/current_cost�,�;�0��+       ��K	XM��A�*

logging/current_cost��;bp:b+       ��K	߲M��A�*

logging/current_cost��;�f3�+       ��K	��M��A�*

logging/current_cost���;����+       ��K	�N��A�*

logging/current_costH�;�x]9+       ��K	5ON��A�*

logging/current_cost#�;�=�^+       ��K	ЁN��A�*

logging/current_cost�ی;/9c$+       ��K	�N��A�*

logging/current_cost�Ќ;�k�)+       ��K	}�N��A�*

logging/current_cost�ƌ;��C�+       ��K	�O��A�*

logging/current_cost޼�;�}V�+       ��K	�ZO��A�*

logging/current_costN��;n�+       ��K	��O��A�*

logging/current_cost;����+       ��K	o�O��A�*

logging/current_cost���;��u+       ��K	��O��A�*

logging/current_costʗ�;F�X+       ��K	�,P��A�*

logging/current_cost鎌;a�w	+       ��K	�aP��A�*

logging/current_cost=��;ь��+       ��K	ݐP��A�*

logging/current_cost�}�;��>�+       ��K	<�P��A�*

logging/current_cost�t�;L��++       ��K	J�P��A�*

logging/current_costql�;����+       ��K	�&Q��A�*

logging/current_costd�;�hD+       ��K	�VQ��A�*

logging/current_cost�[�;��gx+       ��K	��Q��A�*

logging/current_cost\S�;8&��+       ��K	<�Q��A�*

logging/current_cost�J�;[�C6+       ��K	��Q��A�*

logging/current_cost�B�;�^*#+       ��K	�'R��A�*

logging/current_cost=:�;b�D+       ��K	)YR��A�*

logging/current_cost�1�;x��e+       ��K	t�R��A�*

logging/current_cost�)�;�qi�+       ��K	�R��A�*

logging/current_cost�!�;���(+       ��K	��R��A�*

logging/current_cost��;�/��+       ��K	*'S��A�*

logging/current_cost6�;��A+       ��K	]S��A�*

logging/current_cost�
�;e��+       ��K	t�S��A�*

logging/current_cost��;���q+       ��K	 �S��A�*

logging/current_costI��;�}+       ��K	b
T��A�*

logging/current_cost��;����+       ��K	�@T��A�*

logging/current_costJ�;��S+       ��K	RrT��A�*

logging/current_cost��;v��=+       ��K	#�T��A�*

logging/current_cost�݋;��S6+       ��K	3�T��A�*

logging/current_cost+֋;�~��+       ��K	�U��A�*

logging/current_cost�΋;5"<+       ��K	DBU��A�*

logging/current_cost�ǋ;��PD+       ��K	^uU��A�*

logging/current_cost���;R ) +       ��K	9�U��A�*

logging/current_cost���;�R��+       ��K	a�U��A�*

logging/current_cost���;~YT�+       ��K	V��A�*

logging/current_cost���;q���+       ��K	�IV��A�*

logging/current_cost���;Ah�+       ��K	Y~V��A�*

logging/current_cost۝�;M��+       ��K	K�V��A�*

logging/current_cost���;n�N@+       ��K	w�V��A�*

logging/current_cost4��;n ��+       ��K	XW��A�*

logging/current_costk��;>KJj+       ��K	�GW��A�*

logging/current_cost���;��+       ��K	�wW��A�*

logging/current_cost�{�;�0 �+       ��K	/�W��A�*

logging/current_cost1u�;����+       ��K	R�W��A�*

logging/current_cost�n�;j���+       ��K	WX��A�*

logging/current_cost�g�;-��++       ��K	Z;X��A�*

logging/current_costma�;R�ۀ+       ��K	kX��A�*

logging/current_cost[�;��&+       ��K	��X��A�*

logging/current_cost�T�;���l+       ��K	��X��A�*

logging/current_cost�M�;�m�?+       ��K	HY��A�*

logging/current_costkG�;�y��+       ��K	h;Y��A�*

logging/current_cost�@�;�f��+       ��K	7nY��A�*

logging/current_cost�:�;�z�%+       ��K	G�Y��A�*

logging/current_cost�4�;�k�L+       ��K	��Y��A�*

logging/current_cost�.�;���+       ��K	9Z��A�*

logging/current_cost )�;0�w+       ��K	�EZ��A�*

logging/current_cost/#�;&��+       ��K	�zZ��A�*

logging/current_costc�;���7+       ��K	׬Z��A�*

logging/current_cost��;1��+       ��K		�Z��A�*

logging/current_cost�;)_+       ��K	�[��A�*

logging/current_costp�;DL�<+       ��K	�C[��A�*

logging/current_cost��;l+0A+       ��K	x[��A�*

logging/current_costT�;xK*+       ��K	7�[��A�*

logging/current_cost���;�r'X+       ��K	��[��A�*

logging/current_costU�;C�+       ��K	|\��A�*

logging/current_cost4�;+�t+       ��K	\U\��A�*

logging/current_cost��;��f(+       ��K	g�\��A�*

logging/current_cost�ڊ;�o�]+       ��K	ؿ\��A�*

logging/current_cost�ъ;|��6+       ��K	%�\��A�*

logging/current_cost�Ȋ;&��+       ��K	�&]��A�*

logging/current_cost:��;&s+�+       ��K	C\]��A�*

logging/current_cost床;zI�+       ��K	��]��A�*

logging/current_cost���;��+       ��K	��]��A�*

logging/current_cost���;z��+       ��K	��]��A�*

logging/current_cost���; -?0+       ��K	e(^��A�*

logging/current_cost���;�l�p+       ��K	�\^��A�*

logging/current_cost���;`�O�+       ��K	b�^��A�*

logging/current_costҜ�;�%��+       ��K	��^��A�*

logging/current_cost	��;K��+       ��K	��^��A�*

logging/current_costG��;$� k+       ��K	�)_��A�*

logging/current_cost���;���+       ��K	 \_��A�*

logging/current_cost͉�;��;u+       ��K	��_��A�*

logging/current_cost%��;��+       ��K	�_��A�*

logging/current_cost���;l�N]+       ��K	��_��A�*

logging/current_cost�{�;S���+       ��K	+,`��A�*

logging/current_costUw�;%���+       ��K	^`��A�*

logging/current_cost�r�;��f+       ��K	��`��A�*

logging/current_costEn�;mr�a+       ��K	z�`��A�*

logging/current_cost�i�;b�_�+       ��K	��`��A�*

logging/current_costIe�;��*+       ��K	e*a��A�*

logging/current_cost�`�;h\k#+       ��K	Nua��A�*

logging/current_costm\�;�W��+       ��K	�a��A�*

logging/current_costX�;�@�+       ��K	�a��A�*

logging/current_cost�S�;�9��+       ��K	�b��A�*

logging/current_costJO�;nfٜ+       ��K	2Pb��A�*

logging/current_cost�J�;��0�+       ��K	��b��A�*

logging/current_cost�F�;z��%+       ��K	��b��A�	*

logging/current_costSB�;"Вh+       ��K	$c��A�	*

logging/current_cost>�;pVe+       ��K	�Nc��A�	*

logging/current_cost�9�;�u�9+       ��K	��c��A�	*

logging/current_cost�5�;�ĳ +       ��K	�c��A�	*

logging/current_costF1�;^��+       ��K	:�c��A�	*

logging/current_cost-�;�.�a+       ��K	�(d��A�	*

logging/current_cost�(�;�<�m+       ��K	A`d��A�	*

logging/current_cost�$�;I�#+       ��K	�d��A�	*

logging/current_cost� �;tb�/+       ��K	��d��A�	*

logging/current_costh�;-o�t+       ��K	��d��A�	*

logging/current_costd�;�)+       ��K	Z,e��A�	*

logging/current_costK�;��e�+       ��K	�`e��A�	*

logging/current_costS�;�eS(+       ��K	��e��A�	*

logging/current_costT�;�LË+       ��K	_�e��A�	*

logging/current_costP�;Hq[+       ��K	��e��A�	*

logging/current_costg�;�Yh�+       ��K	�,f��A�	*

logging/current_costs �;.�lA+       ��K	�]f��A�	*

logging/current_cost���;���+       ��K	o�f��A�	*

logging/current_cost���;���_+       ��K	^�f��A�	*

logging/current_cost��;��2�+       ��K	5�f��A�	*

logging/current_cost�;K+       ��K	5g��A�	*

logging/current_cost8�;��2J+       ��K	xjg��A�	*

logging/current_cost]�;�uWF+       ��K	�g��A�	*

logging/current_cost9�;DZ�+       ��K	��g��A�	*

logging/current_cost;�;�U�+       ��K	Zh��A�
*

logging/current_costY݉;����+       ��K	�7h��A�
*

logging/current_costcى;��+       ��K	�kh��A�
*

logging/current_costtՉ;�z_+       ��K	��h��A�
*

logging/current_costqщ;! �+       ��K	��h��A�
*

logging/current_cost�͉;�4+       ��K	i��A�
*

logging/current_cost�ɉ;��:�+       ��K	~;i��A�
*

logging/current_cost�ŉ;�uM+       ��K	fpi��A�
*

logging/current_cost���;v�q+       ��K	��i��A�
*

logging/current_costɽ�;cl+       ��K	��i��A�
*

logging/current_cost��;��0�+       ��K	j��A�
*

logging/current_costN��;3*�N+       ��K	�=j��A�
*

logging/current_cost$��;H�!�+       ��K	�nj��A�
*

logging/current_cost���;��+       ��K	ʣj��A�
*

logging/current_costK��;vB�f+       ��K	��j��A�
*

logging/current_cost6��;t1�+       ��K	Yk��A�
*

logging/current_cost�j�;�+       ��K	[Ok��A�
*

logging/current_cost_C�;}j/�+       ��K	f�k��A�
*

logging/current_cost�.�;��V+       ��K	;�k��A�
*

logging/current_cost�!�;�=o�+       ��K	z�k��A�
*

logging/current_cost��;��g0+       ��K	ol��A�
*

logging/current_cost �;r��x+       ��K	�Ql��A�
*

logging/current_cost� �;�+��+       ��K	!�l��A�
*

logging/current_cost��;Z�*:+       ��K	(�l��A�
*

logging/current_cost��;o��+       ��K	�l��A�
*

logging/current_cost/߈;��w�+       ��K	� m��A�
*

logging/current_cost�Ԉ;p�h�+       ��K	ZRm��A�*

logging/current_cost�ʈ;����+       ��K	��m��A�*

logging/current_cost��;����+       ��K	��m��A�*

logging/current_cost-��;�@�2+       ��K	��m��A�*

logging/current_cost���;{�@Y+       ��K	�n��A�*

logging/current_cost���;��A+       ��K	pCn��A�*

logging/current_cost$��;4?�"+       ��K	�tn��A�*

logging/current_costƅ�;�V@�+       ��K	"�n��A�*

logging/current_cost�{�;�E�?+       ��K	z�n��A�*

logging/current_cost�q�;�+       ��K	Ao��A�*

logging/current_cost�g�;���+       ��K	(Fo��A�*

logging/current_cost�^�;���+       ��K	�wo��A�*

logging/current_costzU�;1�+       ��K	өo��A�*

logging/current_cost%L�;��iS+       ��K	��o��A�*

logging/current_costC�;���+       ��K	Dp��A�*

logging/current_cost:�;o�+       ��K	�?p��A�*

logging/current_costb1�;t�WV+       ��K	�rp��A�*

logging/current_cost�(�;��`+       ��K	��p��A�*

logging/current_cost3 �;pE�+       ��K	��p��A�*

logging/current_cost��;]KJB+       ��K	Oq��A�*

logging/current_cost��;���+       ��K	�8q��A�*

logging/current_cost��;�#�!+       ��K	\jq��A�*

logging/current_cost}��;碬
+       ��K	؛q��A�*

logging/current_cost���;�ռ/+       ��K	e�q��A�*

logging/current_cost��;3��n+       ��K	k�q��A�*

logging/current_cost5�;��+       ��K	91r��A�*

logging/current_cost�އ;�i��+       ��K	�ar��A�*

logging/current_cost�ׇ;C���+       ��K	T�r��A�*

logging/current_cost�Ї;X��A+       ��K	��r��A�*

logging/current_costuʇ;�WS+       ��K	e�r��A�*

logging/current_cost+ć;�!��+       ��K	�#s��A�*

logging/current_cost���;Jj�o+       ��K	RXs��A�*

logging/current_cost5��;I8�5+       ��K	��s��A�*

logging/current_cost첇;%2b+       ��K	%�s��A�*

logging/current_cost{��;+��&+       ��K	��s��A�*

logging/current_cost��;�}}+       ��K	+t��A�*

logging/current_costϢ�;\d[+       ��K	�^t��A�*

logging/current_cost:��;'D�+       ��K	�t��A�*

logging/current_cost%��; F\|+       ��K	�t��A�*

logging/current_costK��;����+       ��K	��t��A�*

logging/current_cost���;L���+       ��K		&u��A�*

logging/current_costъ�;{�t�+       ��K	�[u��A�*

logging/current_costr��;��1+       ��K	�u��A�*

logging/current_cost:��;�4��+       ��K	̽u��A�*

logging/current_costE~�;��+       ��K	��u��A�*

logging/current_cost%z�;V�j+       ��K	~&v��A�*

logging/current_costMv�;��v�+       ��K	4Xv��A�*

logging/current_cost�r�;����+       ��K	o�v��A�*

logging/current_cost!o�;�a+       ��K	��v��A�*

logging/current_costDk�;�;o�+       ��K	��v��A�*

logging/current_cost�g�;b'�_+       ��K	
/w��A�*

logging/current_cost+d�;��O+       ��K	dw��A�*

logging/current_cost�`�;����+       ��K	Z�w��A�*

logging/current_costj]�;kf�5+       ��K	��w��A�*

logging/current_costZ�;O�++       ��K	F�w��A�*

logging/current_cost�V�;�(T+       ��K	�'x��A�*

logging/current_cost�S�;J}~Z+       ��K	�Wx��A�*

logging/current_cost_P�;��Q2+       ��K	y�x��A�*

logging/current_costM�;�V6v+       ��K	+�x��A�*

logging/current_costJ�;eZ]�+       ��K	G�x��A�*

logging/current_costG�;y��+       ��K	�y��A�*

logging/current_cost�C�;mD�i+       ��K	�;y��A�*

logging/current_costA�;�=�+       ��K	�hy��A�*

logging/current_cost>�;L�J�+       ��K	�y��A�*

logging/current_costB;�;4v+       ��K	v�y��A�*

logging/current_cost8�;���i+       ��K	��y��A�*

logging/current_cost5�;��!�+       ��K	�"z��A�*

logging/current_cost 2�;���+       ��K	yPz��A�*

logging/current_cost"/�;��!+       ��K	-}z��A�*

logging/current_costw,�;��1+       ��K	��z��A�*

logging/current_cost�)�;R�>X+       ��K	��z��A�*

logging/current_cost�&�;.��+       ��K	8{��A�*

logging/current_cost$�;�g��+       ��K	�7{��A�*

logging/current_cost|!�;x��?+       ��K	�d{��A�*

logging/current_cost��;��>"+       ��K	f�{��A�*

logging/current_cost5�;�N@�+       ��K	�|��A�*

logging/current_cost��;׼>�+       ��K	�<|��A�*

logging/current_cost �;aCK�+       ��K	q�|��A�*

logging/current_cost��;Z���+       ��K	�|��A�*

logging/current_cost�;�%+�+       ��K	��|��A�*

logging/current_cost��;V�Q+       ��K	�"}��A�*

logging/current_cost3�;�e�+       ��K	�X}��A�*

logging/current_cost�
�;GX�+       ��K	Չ}��A�*

logging/current_cost��;�R7+       ��K	D�}��A�*

logging/current_costQ�;���p+       ��K	+�}��A�*

logging/current_cost��; %�K+       ��K	�!~��A�*

logging/current_cost��;8���+       ��K	�X~��A�*

logging/current_costa��; ڲ�+       ��K	v�~��A�*

logging/current_cost��;F~��+       ��K	!�~��A�*

logging/current_cost��;�z�U+       ��K	��~��A�*

logging/current_cost���;kZ�+       ��K	<��A�*

logging/current_cost���;b��+       ��K	DC��A�*

logging/current_cost��;,��+       ��K	�q��A�*

logging/current_costx�;�@!�+       ��K	���A�*

logging/current_costh��;��i+       ��K	����A�*

logging/current_cost_�;KB�+       ��K	9���A�*

logging/current_costg�;7��+       ��K	?���A�*

logging/current_costc�;F<T+       ��K	z|���A�*

logging/current_cost��;�؅+       ��K	�����A�*

logging/current_costk�;����+       ��K	����A�*

logging/current_cost��;�Ni5+       ��K	�B���A�*

logging/current_cost��;P_;+       ��K	){���A�*

logging/current_cost�; �&�+       ��K	w����A�*

logging/current_cost)߆;�Xa�+       ��K	�ၓ�A�*

logging/current_cost�݆;���h+       ��K	���A�*

logging/current_cost�ۆ;�0 �+       ��K	�^���A�*

logging/current_cost#چ;���<+       ��K	1����A�*

logging/current_cost�؆;WVZ�+       ��K	
˂��A�*

logging/current_cost�ֆ;=-�*+       ��K	����A�*

logging/current_costjՆ;%m�+       ��K	>���A�*

logging/current_cost�ӆ;��z+       ��K	vy���A�*

logging/current_costs҆;��+       ��K	4����A�*

logging/current_costц;:W�+       ��K	f僓�A�*

logging/current_costfφ;C��+       ��K	����A�*

logging/current_costΆ;��=k+       ��K	�P���A�*

logging/current_cost�̆;W���+       ��K	M����A�*

logging/current_cost(ˆ;�ۆ+       ��K	O����A�*

logging/current_cost�Ɇ;c��>+       ��K	g焓�A�*

logging/current_costcȆ;�d�+       ��K	�����A�*

logging/current_costǆ;�؎�+       ��K	1����A�*

logging/current_cost�ņ;�MM+       ��K	4�A�*

logging/current_costrĆ;���+       ��K	�+���A�*

logging/current_costÆ;�E�N+       ��K	`���A�*

logging/current_cost���;�f�=+       ��K	j����A�*

logging/current_cost���;idO7+       ��K	F͆��A�*

logging/current_costO��;���7+       ��K	����A�*

logging/current_cost˽�;��+       ��K	�7���A�*

logging/current_cost���;8ݮ+       ��K	�d���A�*

logging/current_costX��;n��+       ��K	瓇��A�*

logging/current_cost*��;L��+       ��K	n����A�*

logging/current_cost޸�;�r�:+       ��K	��A�*

logging/current_cost���;�
�+       ��K	3���A�*

logging/current_cost���;� �+       ��K	�K���A�*

logging/current_costq��;���+       ��K	�x���A�*

logging/current_cost1��;㫖+       ��K	$����A�*

logging/current_cost!��;�@�G+       ��K	�݈��A�*

logging/current_cost��;3���+       ��K	����A�*

logging/current_cost���; �+       ��K	�9���A�*

logging/current_costݯ�;c��&+       ��K	uh���A�*

logging/current_cost���;ߵ�>+       ��K	
����A�*

logging/current_cost���;�n#�+       ��K	tŉ��A�*

logging/current_cost���;��+       ��K	x��A�*

logging/current_cost���;x��K+       ��K	{#���A�*

logging/current_costS��;Юc+       ��K	cP���A�*

logging/current_costJ��;R��+       ��K	�}���A�*

logging/current_cost\��;�)��+       ��K	�����A�*

logging/current_cost,��;�l�+       ��K	�ي��A�*

logging/current_cost��;�9�+       ��K	U	���A�*

logging/current_cost餆;L�ƾ+       ��K	7���A�*

logging/current_cost��;τ�+       ��K	�e���A�*

logging/current_cost���;fwX+       ��K	�����A�*

logging/current_cost���;O�Z�+       ��K	ċ��A�*

logging/current_cost-��;K��+       ��K	���A�*

logging/current_cost@��;`(�+       ��K	$���A�*

logging/current_costu��;�/t�+       ��K	�Y���A�*

logging/current_costZ��;[��+       ��K	Չ���A�*

logging/current_cost@��;�L�6+       ��K	�����A�*

logging/current_costk��;h��+       ��K	�挓�A�*

logging/current_cost~��;g+       ��K	� ���A�*

logging/current_cost���; ?W!+       ��K	NT���A�*

logging/current_costÙ�;*ˎq+       ��K	炍��A�*

logging/current_costט�;�Q�+       ��K	ϵ���A�*

logging/current_cost֗�;n!R�+       ��K	�卓�A�*

logging/current_cost;��]_+       ��K	-���A�*

logging/current_cost���;�b�+       ��K	�F���A�*

logging/current_cost��;�=��+       ��K	bx���A�*

logging/current_costד�;鑞�+       ��K	m����A�*

logging/current_costĒ�;}��5+       ��K	QҎ��A�*

logging/current_costӑ�;㉼�+       ��K	o���A�*

logging/current_costӐ�;�f�+       ��K	�1���A�*

logging/current_cost폆;4?q#+       ��K	�b���A�*

logging/current_cost��;t�I�+       ��K	�����A�*

logging/current_cost-��;j�+       ��K	b����A�*

logging/current_cost6��;��+       ��K	�鏓�A�*

logging/current_costf��;��/o+       ��K	����A�*

logging/current_costU��;~{k+       ��K	H���A�*

logging/current_cost��; g�/+       ��K	;x���A�*

logging/current_cost���;�'+       ��K	D����A�*

logging/current_costЈ�;60K +       ��K	pא��A�*

logging/current_cost燆;�3B+       ��K	9���A�*

logging/current_cost(��;�75s+       ��K	�4���A�*

logging/current_costb��;o�o�+       ��K	De���A�*

logging/current_cost���;5�B+       ��K	ߕ���A�*

logging/current_cost���;%��Q+       ��K	���A�*

logging/current_cost���;��:�+       ��K	��A�*

logging/current_costP��;�[=s+       ��K	{ ���A�*

logging/current_costS�;%i��+       ��K	TO���A�*

logging/current_costS~�;]}C�+       ��K	�|���A�*

logging/current_costy}�;O���+       ��K	�����A�*

logging/current_costq|�;S��+       ��K	iݒ��A�*

logging/current_cost�{�;���+       ��K	
���A�*

logging/current_cost�z�;�ҷ�+       ��K	�:���A�*

logging/current_cost8y�;�QV+       ��K	0h���A�*

logging/current_costx�;����+       ��K	!����A�*

logging/current_cost�v�;|V�+       ��K	�̓��A�*

logging/current_cost�u�;K��+       ��K	D����A�*

logging/current_cost�t�;���+       ��K	,���A�*

logging/current_cost�s�;�9+       ��K	sY���A�*

logging/current_cost�r�;C�@�+       ��K	����A�*

logging/current_cost�q�;�/�P+       ��K	�����A�*

logging/current_cost�p�;�XQv+       ��K	唓�A�*

logging/current_cost�o�;Lڒ+       ��K	����A�*

logging/current_cost�n�;&f�+       ��K	�=���A�*

logging/current_cost#n�;zh6+       ��K	dm���A�*

logging/current_costfm�;��m+       ��K	盕��A�*

logging/current_cost�l�;��,+       ��K	�ʕ��A�*

logging/current_cost�k�;�%+       ��K	+����A�*

logging/current_cost�j�;��+       ��K	�#���A�*

logging/current_cost�i�;O�n+       ��K	�N���A�*

logging/current_cost�h�;��0+       ��K	�|���A�*

logging/current_costh�;�m�F+       ��K	<����A�*

logging/current_cost$g�;��vm+       ��K	�ז��A�*

logging/current_cost[f�;�rl+       ��K	����A�*

logging/current_costme�;��xI+       ��K	P/���A�*

logging/current_cost�d�;9'+       ��K	�Z���A�*

logging/current_cost�c�;��+       ��K	�����A�*

logging/current_cost)c�;`�B�+       ��K	����A�*

logging/current_cost\b�;y�4�+       ��K	v藓�A�*

logging/current_costja�;�c5�+       ��K	����A�*

logging/current_costw`�;5�+       ��K	?���A�*

logging/current_cost�_�;���#+       ��K	�l���A�*

logging/current_cost�^�;+ڐ�+       ��K	�����A�*

logging/current_cost^�; �
.+       ��K	�Ř��A�*

logging/current_cost+]�;�eV�+       ��K	!��A�*

logging/current_costU\�;V�[1+       ��K	v���A�*

logging/current_cost�[�;t���+       ��K	�O���A�*

logging/current_cost�Z�;,Tr�+       ��K	����A�*

logging/current_cost�Y�;�y��+       ��K	�����A�*

logging/current_cost�X�;8TMw+       ��K	ܙ��A�*

logging/current_cost*X�;"0��+       ��K	����A�*

logging/current_costNW�;�AOW+       ��K	7���A�*

logging/current_costrV�;�z�+       ��K	Ad���A�*

logging/current_cost�U�;ҍp+       ��K	̑���A�*

logging/current_cost�Q�;�l$H+       ��K	7Ś��A�*

logging/current_cost�@�;Ͷ�+       ��K	����A�*

logging/current_costm(�; W++       ��K	�"���A�*

logging/current_cost�;$p+       ��K	;O���A�*

logging/current_cost<�;����+       ��K	s|���A�*

logging/current_cost'ۅ;�F�7+       ��K	����A�*

logging/current_costzÅ;4��+       ��K	6����A�*

logging/current_costث�;FsS+       ��K	����A�*

logging/current_costT��;n���+       ��K	A���A�*

logging/current_cost$��;̠��+       ��K	�t���A�*

logging/current_cost���;���+       ��K	Z����A�*

logging/current_cost<�;_&�+       ��K	�Ҝ��A�*

logging/current_cost&{�;:��+       ��K	����A�*

logging/current_cost)w�;2{�+       ��K	f.���A�*

logging/current_costs�;�=��+       ��K	X\���A�*

logging/current_cost�n�;u`i�+       ��K	�����A�*

logging/current_cost�j�;�3��+       ��K	�����A�*

logging/current_costug�;�Lq+       ��K	�읓�A�*

logging/current_cost�c�;�Hò+       ��K	%���A�*

logging/current_cost;`�;^�k+       ��K	�F���A�*

logging/current_cost�\�;���+       ��K	Y}���A�*

logging/current_costY�;ͩr�+       ��K	]����A�*

logging/current_cost�U�;�O$+       ��K	!ٞ��A�*

logging/current_costR�;��+       ��K	P���A�*

logging/current_cost�N�;�=+       ��K	l7���A�*

logging/current_cost�K�;l�щ+       ��K	e���A�*

logging/current_costoH�;�M�+       ��K	�����A�*

logging/current_cost�E�;��w<+       ��K	2ҟ��A�*

logging/current_cost�B�;�r"�+       ��K	����A�*

logging/current_cost{?�;�&�,+       ��K	�.���A�*

logging/current_cost�<�;s�+       ��K	�Z���A�*

logging/current_cost�9�;GK��+       ��K	4����A�*

logging/current_cost7�;�Po+       ��K	滠��A�*

logging/current_cost
4�;�BG}+       ��K	�ꠓ�A�*

logging/current_cost11�;��_S+       ��K	c���A�*

logging/current_cost_.�;�T�+       ��K	�E���A�*

logging/current_cost�+�;+4'+       ��K	s���A�*

logging/current_cost�(�;A�m+       ��K	f����A�*

logging/current_costU&�;J��!+       ��K	�С��A�*

logging/current_cost�#�;bd��+       ��K	����A�*

logging/current_cost@!�;J���+       ��K	*���A�*

logging/current_cost��;��+       ��K	@W���A�*

logging/current_cost�;<�t+       ��K	T����A�*

logging/current_cost��;��v+       ��K	Ư���A�*

logging/current_cost��;d�B<+       ��K	ޢ��A�*

logging/current_cost&�;�}�/+       ��K	

���A�*

logging/current_cost��;��t3+       ��K	.9���A�*

logging/current_costz�;��"+       ��K	�g���A�*

logging/current_cost+�;K�/+       ��K	�����A�*

logging/current_cost�;#S
�+       ��K	�£��A�*

logging/current_costP�;?��*+       ��K	��A�*

logging/current_cost�;=/�+       ��K	����A�*

logging/current_cost��;2�.�+       ��K	=N���A�*

logging/current_cost
�;^��+       ��K	�|���A�*

logging/current_cost ��;A5��+       ��K	T����A�*

logging/current_cost���;��?_+       ��K	�֤��A�*

logging/current_cost���;K��+       ��K	����A�*

logging/current_costd��;cj�3+       ��K	$0���A�*

logging/current_cost��;�I�+       ��K	P^���A�*

logging/current_cost�;�,#+       ��K	$����A�*

logging/current_cost��;�D�+       ��K	�����A�*

logging/current_cost��;Ss`(+       ��K	�奓�A�*

logging/current_cost3�;z��+       ��K	���A�*

logging/current_cost��;�V�G+       ��K	C���A�*

logging/current_cost,�;�O{h+       ��K	+p���A�*

logging/current_cost2�;K?�+       ��K	����A�*

logging/current_cost�݄;'ʹ&+       ��K	�ͦ��A�*

logging/current_costۄ;^�5�+       ��K	o����A�*

logging/current_cost6؄;85�a+       ��K	�*���A�*

logging/current_cost�Մ;7M�+       ��K	�[���A�*

logging/current_cost�҄;��l�+       ��K	,����A�*

logging/current_costЄ;~�i+       ��K	����A�*

logging/current_cost�̈́;0�xu+       ��K	�����A�*

logging/current_costY˄;>9+       ��K	���A�*

logging/current_cost�Ȅ;+�+       ��K	7K���A�*

logging/current_cost�Ƅ;k��&+       ��K	�x���A�*

logging/current_costĄ;���+       ��K	$����A�*

logging/current_cost&;
U'+       ��K	�֨��A�*

logging/current_costƿ�;>���+       ��K	����A�*

logging/current_cost���;�+       ��K	2���A�*

logging/current_cost<��;�1�~+       ��K	�c���A�*

logging/current_costǸ�;��!C+       ��K	c����A�*

logging/current_costg��;�l��+       ��K	俩��A�*

logging/current_costճ�;�`.+       ��K	"禎�A�*

logging/current_costd��;"W�+       ��K	! ���A�*

logging/current_cost���;��v�+       ��K	�M���A�*

logging/current_costӭ�;����+       ��K	Zz���A�*

logging/current_costy��;{�+       ��K	\����A�*

logging/current_costT��;H�~^+       ��K	�۪��A�*

logging/current_cost[��;>��6+       ��K	
���A�*

logging/current_costA��;����+       ��K	�8���A�*

logging/current_cost���;�@I+       ��K	�d���A�*

logging/current_costX��;��0�+       ��K	�����A�*

logging/current_costA��;!v�=+       ��K	�ū��A�*

logging/current_cost���;@i@6+       ��K	���A�*

logging/current_costț�;f�G�+       ��K	�"���A�*

logging/current_costę�;���I+       ��K	�V���A�*

logging/current_cost���;��f+       ��K	%����A�*

logging/current_cost땄;wС+       ��K	�����A�*

logging/current_cost���;� �+       ��K	�ଓ�A�*

logging/current_cost⒄;��7+       ��K	u���A�*

logging/current_costJ��;(���+       ��K	L<���A�*

logging/current_cost+��;�_+       ��K	�j���A�*

logging/current_cost��;��EA+       ��K	Ҙ���A�*

logging/current_cost֋�;�H�+       ��K	�ʭ��A�*

logging/current_costO��;*m+       ��K	|����A�*

logging/current_cost��;ۗ�+       ��K	%'���A�*

logging/current_cost놄; �m+       ��K	W���A�*

logging/current_costI��;bw��+       ��K		����A�*

logging/current_cost���;Ҥҧ+       ��K	�����A�*

logging/current_cost���;I���+       ��K	�ޮ��A�*

logging/current_cost���;m3��+       ��K	����A�*

logging/current_costE�;P3�+       ��K	=���A�*

logging/current_cost}�;T��+       ��K	�k���A�*

logging/current_cost�{�;K-|�+       ��K	A����A�*

logging/current_cost z�;p9^X+       ��K	ʯ��A�*

logging/current_cost�x�;}Dy+       ��K	����A�*

logging/current_costlw�;_���+       ��K	�%���A�*

logging/current_cost�u�;u�v+       ��K	JV���A�*

logging/current_cost't�;�� �+       ��K	����A�*

logging/current_costs�;�B>�+       ��K	�����A�*

logging/current_costyq�;h�!�+       ��K	�ݰ��A�*

logging/current_costp�;�(*\+       ��K	b���A�*

logging/current_cost�n�;3$0+       ��K	9���A�*

logging/current_cost�l�;_��+       ��K	�f���A�*

logging/current_costuk�;�8-�+       ��K	ӓ���A�*

logging/current_cost�i�;X���+       ��K	�����A�*

logging/current_cost�g�;Q��+       ��K	v����A�*

logging/current_cost�f�;��q�+       ��K	v���A�*

logging/current_cost�e�;��+       ��K	�H���A�*

logging/current_cost:d�;l�+       ��K	 v���A�*

logging/current_cost�b�;|@%�+       ��K	I����A�*

logging/current_cost�`�;O=��+       ��K	xҲ��A�*

logging/current_cost�_�;>��+       ��K	�����A�*

logging/current_costM^�;�{\+       ��K	]+���A�*

logging/current_costA]�;Gss+       ��K	 [���A�*

logging/current_cost�[�;�?�+       ��K	�����A�*

logging/current_cost5Z�;�|%+       ��K	!����A�*

logging/current_cost_X�;�m�+       ��K	[೓�A�*

logging/current_costPW�;�7�+       ��K	����A�*

logging/current_cost�U�;�WZ�+       ��K	�=���A�*

logging/current_cost�T�;�Z�+       ��K	�k���A�*

logging/current_costKS�;�SY+       ��K	$����A�*

logging/current_costoQ�;�<))+       ��K	XŴ��A�*

logging/current_cost�O�;�Rt|+       ��K	.����A�*

logging/current_cost�N�;�W�+       ��K	�*���A�*

logging/current_costqM�;�|�2+       ��K	�Z���A�*

logging/current_cost�K�;$��+       ��K	}����A�*

logging/current_costFK�;���)+       ��K	t����A�*

logging/current_cost�H�;7$�(+       ��K	jⵓ�A�*

logging/current_cost&H�;����+       ��K	����A�*

logging/current_cost�F�;ea��+       ��K	?���A�*

logging/current_cost|E�;�J�+       ��K	�q���A�*

logging/current_cost0D�;���|+       ��K	�����A�*

logging/current_cost�B�;��*+       ��K	�ж��A�*

logging/current_cost�@�;��6+       ��K	����A�*

logging/current_cost�?�;�}ؗ+       ��K	�+���A�*

logging/current_cost�=�;�G�+       ��K	-Z���A�*

logging/current_cost=�;���+       ��K	J����A�*

logging/current_cost1;�;JF~x+       ��K	o����A�*

logging/current_cost;:�;�Y��+       ��K	�뷓�A�*

logging/current_cost�8�;��1�+       ��K	����A�*

logging/current_cost�7�;3�\+       ��K	�H���A�*

logging/current_costh6�;|��+       ��K	%v���A�*

logging/current_cost5�;���+       ��K	٦���A�*

logging/current_cost�3�;n৮+       ��K	<ո��A�*

logging/current_cost3�;c�+       ��K	���A�*

logging/current_cost�1�;�	�/+       ��K	�0���A�*

logging/current_cost�/�;�G"�+       ��K	_���A�*

logging/current_costj.�;��=�+       ��K	�����A�*

logging/current_costS-�;���(+       ��K	�����A�*

logging/current_cost�+�;�b+       ��K	�湓�A�*

logging/current_cost+�;l��+       ��K	@���A�*

logging/current_cost�)�;k���+       ��K	C���A�*

logging/current_costU(�;:_`+       ��K	p���A�*

logging/current_costD&�;�j6+       ��K	9����A�*

logging/current_cost�$�;�y�+       ��K	mк��A�*

logging/current_cost�#�;�IQ+       ��K	F ���A�*

logging/current_cost#�;����+       ��K	�.���A�*

logging/current_cost{!�;��p+       ��K	c[���A�*

logging/current_costz �;J�i@+       ��K	v����A�*

logging/current_costH�;�俰+       ��K	���A�*

logging/current_cost1�;�س+       ��K	>*���A�*

logging/current_cost��;�G%D+       ��K	�f���A�*

logging/current_cost�;�(1+       ��K	�����A�*

logging/current_costx�;Xr�+       ��K	�ܼ��A�*

logging/current_cost-�;�.�5+       ��K	����A�*

logging/current_cost@�;�T+       ��K	dY���A�*

logging/current_cost��;�
'7+       ��K	L����A�*

logging/current_cost/�;>V�>+       ��K	pҽ��A�*

logging/current_costI�;��+       ��K	����A�*

logging/current_cost��;�`2+       ��K	&G���A�*

logging/current_costr�;��:+       ��K	����A�*

logging/current_cost>�;ܨ�+       ��K	 ����A�*

logging/current_cost��;;�e�+       ��K	���A�*

logging/current_cost��;�M��+       ��K	�,���A�*

logging/current_costG�;���
+       ��K	lb���A�*

logging/current_cost�	�;���+       ��K	󑿓�A�*

logging/current_cost
	�;ԭ+       ��K	�˿��A�*

logging/current_costi�;5��+       ��K	����A�*

logging/current_cost��;��+       ��K	�2���A�*

logging/current_costA�;�y��+       ��K	Ic���A�*

logging/current_cost��;_�m+       ��K	����A�*

logging/current_cost��;ѩ�+       ��K	V����A�*

logging/current_costE�;Q�Sp+       ��K	�����A�*

logging/current_coste �;�b0+       ��K	0)���A�*

logging/current_cost���;_�}+       ��K	/W���A�*

logging/current_cost���;N�4+       ��K	�����A�*

logging/current_cost���;Wls+       ��K	�����A�*

logging/current_costG��;��֫+       ��K	�����A�*

logging/current_cost���;�X�+       ��K	q�A�*

logging/current_costT��;|l!�+       ��K	Q�A�*

logging/current_cost��;�Ŵ+       ��K	Ӏ�A�*

logging/current_cost���;��F�+       ��K	��A�*

logging/current_cost��;��T�+       ��K	,��A�*

logging/current_costU�;�OIf+       ��K	�Ó�A�*

logging/current_costX�;���p+       ��K	#LÓ�A�*

logging/current_cost���;��b
+       ��K	ZzÓ�A�*

logging/current_cost��;��q�+       ��K	
�Ó�A�*

logging/current_cost��;�a��+       ��K	��Ó�A�*

logging/current_cost �;þ��+       ��K	�ē�A�*

logging/current_cost��;t N+       ��K	�<ē�A�*

logging/current_cost��;6j��+       ��K	�mē�A�*

logging/current_costi�;�V��+       ��K	��ē�A�*

logging/current_cost �;ܪ�|+       ��K	�ē�A�*

logging/current_cost5�;��+       ��K	)�ē�A�*

logging/current_cost��;�[��+       ��K	")œ�A�*

logging/current_cost��;��7�+       ��K	Wœ�A�*

logging/current_cost��;�R��+       ��K	(�œ�A�*

logging/current_cost��;�}Wz+       ��K	�œ�A�*

logging/current_cost,�;�Rԑ+       ��K	��œ�A�*

logging/current_cost�߃;L�R�+       ��K	'"Ɠ�A�*

logging/current_cost�ރ;��	�+       ��K	�RƓ�A�*

logging/current_costH݃;�uV�+       ��K	��Ɠ�A�*

logging/current_costq܃;͂�+       ��K	k�Ɠ�A�*

logging/current_cost�ڃ;�7=+       ��K	��Ɠ�A�*

logging/current_cost�ك;�K�c+       ��K	�Ǔ�A�*

logging/current_cost�؃;�f+       ��K	n>Ǔ�A�*

logging/current_cost[׃;iIX+       ��K	zkǓ�A�*

logging/current_cost�Ճ;���+       ��K	o�Ǔ�A�*

logging/current_cost�ԃ;v]�+       ��K	d�Ǔ�A�*

logging/current_costKӃ;ҥ��+       ��K	��Ǔ�A�*

logging/current_costH҃;4�+       ��K	I&ȓ�A�*

logging/current_cost*у;_��+       ��K	Sȓ�A�*

logging/current_cost�σ;��+       ��K	��ȓ�A�*

logging/current_cost�΃;K�g�+       ��K	��ȓ�A�*

logging/current_costu̓;a]`+       ��K	f�ȓ�A�*

logging/current_cost�̃;rŧ�+       ��K	�ɓ�A�*

logging/current_cost�ʃ;���x+       ��K	�Bɓ�A�*

logging/current_cost�Ƀ;�msP+       ��K	�qɓ�A�*

logging/current_cost�ȃ;�v��+       ��K	�ɓ�A�*

logging/current_costǃ;�ǈ�+       ��K	��ɓ�A�*

logging/current_costƃ;��x\+       ��K	b�ɓ�A�*

logging/current_cost�ă;u�(�+       ��K	�+ʓ�A�*

logging/current_cost�Ã;6@�+       ��K	�Wʓ�A�*

logging/current_cost�;,��+       ��K	ׄʓ�A�*

logging/current_cost
;��C�+       ��K	ųʓ�A�*

logging/current_cost���;D�]+       ��K	E�ʓ�A�*

logging/current_cost���;0�/+       ��K	�˓�A�*

logging/current_cost7��;��M+       ��K	1=˓�A�*

logging/current_cost���;�� �+       ��K	"i˓�A�*

logging/current_costw��;�D�+       ��K	ȕ˓�A�*

logging/current_costm��;�_�+       ��K	��˓�A�*

logging/current_cost	��;���+       ��K	��˓�A�*

logging/current_cost᷃;�?ߊ+       ��K	]%̓�A�*

logging/current_costͶ�;�:.?+       ��K	�U̓�A�*

logging/current_cost���;��%V+       ��K	��̓�A�*

logging/current_cost���;�l:D+       ��K	�̓�A�*

logging/current_cost3��;���X+       ��K	��̓�A�*

logging/current_cost+��;P�f+       ��K	�"͓�A�*

logging/current_cost߰�;^�R�+       ��K	-^͓�A�*

logging/current_cost��;I+��+       ��K	�͓�A�*

logging/current_cost�;$��b+       ��K	2�͓�A�*

logging/current_costb��;:���+       ��K	{Γ�A�*

logging/current_cost6��;�Ou+       ��K	�XΓ�A�*

logging/current_costa��;`V�+       ��K	/�Γ�A�*

logging/current_cost��;{մ�+       ��K	��Γ�A�*

logging/current_cost���;���+       ��K	�ϓ�A�*

logging/current_costӧ�;��!�+       ��K	�@ϓ�A�*

logging/current_cost���;�v&7+       ��K	(sϓ�A�*

logging/current_cost��;�Y�+       ��K	A�ϓ�A�*

logging/current_cost��;}$�M+       ��K	��ϓ�A�*

logging/current_cost>��;�PK	+       ��K	�Г�A�*

logging/current_cost��;6�J++       ��K	3NГ�A�*

logging/current_cost蠃;9%%)+       ��K	%�Г�A�*

logging/current_cost���;7I��+       ��K	��Г�A�*

logging/current_cost䞃;3�ވ+       ��K	��Г�A�*

logging/current_cost���;v ^3+       ��K	� ѓ�A�*

logging/current_costR��;5|x�+       ��K	Ofѓ�A�*

logging/current_costH��;���+       ��K	_�ѓ�A�*

logging/current_cost���;6�+       ��K	��ѓ�A�*

logging/current_costޘ�;0I��+       ��K	��ѓ�A�*

logging/current_cost㗃;����+       ��K	"1ғ�A�*

logging/current_cost���;�T�+       ��K	�cғ�A�*

logging/current_cost���;¸�+       ��K	��ғ�A�*

logging/current_coste��;�8>�+       ��K	��ғ�A�*

logging/current_cost���;Ө-p+       ��K	�ӓ�A�*

logging/current_cost���;���g+       ��K	w6ӓ�A�*

logging/current_cost;:J�0+       ��K	#mӓ�A�*

logging/current_cost���;ÿ�=+       ��K	.�ӓ�A�*

logging/current_cost/��;:���+       ��K	ԓ�A�*

logging/current_cost;RF��+       ��K	+5ԓ�A�*

logging/current_cost���;r�mY+       ��K	�bԓ�A�*

logging/current_costM��;x��Y+       ��K	׎ԓ�A�*

logging/current_cost9��;�(�+       ��K	B�ԓ�A�*

logging/current_cost���;���}+       ��K	��ԓ�A� *

logging/current_cost�;8k(g+       ��K	lՓ�A� *

logging/current_cost;}eT+       ��K	HKՓ�A� *

logging/current_costڅ�;*�+       ��K	��Փ�A� *

logging/current_cost���;D+       ��K	y�Փ�A� *

logging/current_cost˃�;r��+       ��K	�Փ�A� *

logging/current_cost���;�݊�+       ��K	'֓�A� *

logging/current_costI��;pF�+       ��K	9c֓�A� *

logging/current_cost.��;gf�]+       ��K	F�֓�A� *

logging/current_cost1�;���+       ��K	�֓�A� *

logging/current_cost%~�;R��#+       ��K	& ד�A� *

logging/current_coste}�;��H;+       ��K	s1ד�A� *

logging/current_cost�{�;�πF+       ��K	Ijד�A� *

logging/current_cost�z�;�yR�+       ��K	'�ד�A� *

logging/current_cost�y�;��2+       ��K	�ד�A� *

logging/current_cost�x�;y.��+       ��K	��ד�A� *

logging/current_cost�w�; E��+       ��K	K0ؓ�A� *

logging/current_costv�;4
��+       ��K	�^ؓ�A� *

logging/current_costpu�;E�D�+       ��K	��ؓ�A� *

logging/current_cost�t�;yv��+       ��K	z�ؓ�A� *

logging/current_cost�s�;��zK+       ��K	��ؓ�A� *

logging/current_costr�;����+       ��K	pٓ�A� *

logging/current_cost�p�;Q#q-+       ��K	_Nٓ�A� *

logging/current_costfo�;ӂ*e+       ��K	Ԅٓ�A� *

logging/current_cost4n�;F���+       ��K	O�ٓ�A� *

logging/current_costCm�;�<��+       ��K	#�ٓ�A�!*

logging/current_cost{l�;x��+       ��K	�ړ�A�!*

logging/current_cost%k�;�֫I+       ��K	gIړ�A�!*

logging/current_cost�i�;g��*+       ��K	Izړ�A�!*

logging/current_cost�h�;K��+       ��K	��ړ�A�!*

logging/current_cost�g�;̭"�+       ��K	��ړ�A�!*

logging/current_costvf�;Ż�+       ��K	�ۓ�A�!*

logging/current_cost{e�;HD4�+       ��K	=4ۓ�A�!*

logging/current_costhd�;΅3�+       ��K	�aۓ�A�!*

logging/current_cost2c�;b�م+       ��K	�ۓ�A�!*

logging/current_cost6b�;��:�+       ��K	��ۓ�A�!*

logging/current_cost�`�;o,�+       ��K	[�ۓ�A�!*

logging/current_cost�_�;� �+       ��K	�*ܓ�A�!*

logging/current_cost�^�;�>�o+       ��K	%[ܓ�A�!*

logging/current_cost�]�;u��+       ��K	Čܓ�A�!*

logging/current_cost�\�;:��+       ��K	4�ܓ�A�!*

logging/current_cost[�; �_�+       ��K	��ܓ�A�!*

logging/current_cost[�;Q�L�+       ��K	�ݓ�A�!*

logging/current_cost�X�;%Z�	+       ��K	NHݓ�A�!*

logging/current_costX�;�i�+       ��K	�wݓ�A�!*

logging/current_cost�V�;EF�J+       ��K	��ݓ�A�!*

logging/current_cost�U�;ش��+       ��K	��ݓ�A�!*

logging/current_cost�T�;S��+       ��K	�ޓ�A�!*

logging/current_costTS�;m�PO+       ��K	a3ޓ�A�!*

logging/current_cost�Q�;_Q��+       ��K	Wcޓ�A�!*

logging/current_cost}Q�;�;��+       ��K	.�ޓ�A�!*

logging/current_cost�O�;Ж�+       ��K	��ޓ�A�"*

logging/current_cost<O�;3���+       ��K	��ޓ�A�"*

logging/current_cost0N�;��+       ��K	
 ߓ�A�"*

logging/current_costjL�;��_�+       ��K	6Lߓ�A�"*

logging/current_costZK�;�?U�+       ��K	�ߓ�A�"*

logging/current_cost�J�;|v��+       ��K	��ߓ�A�"*

logging/current_costI�;[C�Z+       ��K	w�ߓ�A�"*

logging/current_cost�G�;g�ů+       ��K	n���A�"*

logging/current_cost�F�;h�@�+       ��K	�<���A�"*

logging/current_cost�E�;���+       ��K	�j���A�"*

logging/current_cost�D�;��z�+       ��K	�����A�"*

logging/current_cost:C�;�S�f+       ��K	P����A�"*

logging/current_cost:B�;ۉ5�+       ��K	�����A�"*

logging/current_cost�@�;��f�+       ��K	�*��A�"*

logging/current_cost�?�;ɒ+       ��K	3X��A�"*

logging/current_cost�>�;,���+       ��K	����A�"*

logging/current_cost=�;,�+       ��K	����A�"*

logging/current_cost�;�;�R�+       ��K	����A�"*

logging/current_costf:�;��H+       ��K	���A�"*

logging/current_cost<9�;W�
+       ��K	iA��A�"*

logging/current_cost�7�;�{+       ��K	n��A�"*

logging/current_cost56�;��w+       ��K	ٜ��A�"*

logging/current_cost�4�;e���+       ��K	s���A�"*

logging/current_cost_3�;m�#�+       ��K	����A�"*

logging/current_costT2�;��+       ��K	�%��A�"*

logging/current_cost�0�;��qH+       ��K	�R��A�#*

logging/current_costz/�;E���+       ��K	y���A�#*

logging/current_cost,.�;(��+       ��K	Q���A�#*

logging/current_costj-�;�*�+       ��K	����A�#*

logging/current_cost�+�;���1+       ��K	
	��A�#*

logging/current_cost*�;��a�+       ��K	76��A�#*

logging/current_cost)�;�+       ��K	|e��A�#*

logging/current_cost�'�;���+       ��K	���A�#*

logging/current_cost'�;��+	+       ��K	L���A�#*

logging/current_costb%�;%��+       ��K	���A�#*

logging/current_cost!$�;�s�+       ��K	�#��A�#*

logging/current_cost�"�;�WY�+       ��K	�Q��A�#*

logging/current_cost"�;��j#+       ��K	����A�#*

logging/current_cost� �;�k��+       ��K	����A�#*

logging/current_cost��;]c�V+       ��K	����A�#*

logging/current_cost�;���+       ��K	�
��A�#*

logging/current_cost��;S�-�+       ��K	�7��A�#*

logging/current_cost��;&�(�+       ��K	�e��A�#*

logging/current_cost�;d��+       ��K	Ò��A�#*

logging/current_cost��;0�+       ��K	����A�#*

logging/current_costK�;ge�+       ��K	���A�#*

logging/current_cost&�;�'�+       ��K	� ��A�#*

logging/current_cost\�;�C$#+       ��K	�O��A�#*

logging/current_cost3�;�P/�+       ��K	l��A�#*

logging/current_cost�;��F+       ��K	6���A�#*

logging/current_cost��;ٻQ+       ��K	����A�#*

logging/current_cost��;C�7+       ��K	�
��A�$*

logging/current_cost)�;�s+       ��K	>7��A�$*

logging/current_costu�;��W�+       ��K	�e��A�$*

logging/current_cost��;��3+       ��K	|���A�$*

logging/current_cost��;���+       ��K	����A�$*

logging/current_cost�
�;Xs$�+       ��K	����A�$*

logging/current_costB	�;�:�B+       ��K	H"��A�$*

logging/current_cost��;s�q�+       ��K	�P��A�$*

logging/current_cost��;MB�+       ��K	&��A�$*

logging/current_cost��;_ՙ:+       ��K	����A�$*

logging/current_cost��;����+       ��K	����A�$*

logging/current_cost��;9�{{+       ��K	���A�$*

logging/current_costj�;��X+       ��K	Q;��A�$*

logging/current_costG�;��l+       ��K	�g��A�$*

logging/current_cost �;nC�+       ��K	c���A�$*

logging/current_cost ��;x�yg+       ��K	���A�$*

logging/current_cost���;0��>+       ��K	w���A�$*

logging/current_cost���;��vE+       ��K	�$��A�$*

logging/current_costy��;���+       ��K	@T��A�$*

logging/current_cost���;���+       ��K	p���A�$*

logging/current_costb��;ܺ�Z+       ��K	���A�$*

logging/current_cost1��;�yn+       ��K	7���A�$*

logging/current_costj��;.��0+       ��K	���A�$*

logging/current_costF��;�&�E+       ��K	*C��A�$*

logging/current_cost��;"�
�+       ��K	�p��A�$*

logging/current_cost��;��wa+       ��K	����A�$*

logging/current_cost�;X�Y�+       ��K	����A�%*

logging/current_cost��;x��}+       ��K	y���A�%*

logging/current_cost���;�I��+       ��K		-��A�%*

logging/current_cost��;g���+       ��K	[��A�%*

logging/current_cost&�;����+       ��K	����A�%*

logging/current_cost)�;�Z�<+       ��K	����A�%*

logging/current_cost�;ֺ��+       ��K	����A�%*

logging/current_costh�;���+       ��K	|��A�%*

logging/current_cost��;���+       ��K	�E��A�%*

logging/current_costu�;h��+       ��K	�s��A�%*

logging/current_costF�;��p+       ��K	)���A�%*

logging/current_cost��; ���+       ��K	����A�%*

logging/current_cost��;�~�k+       ��K	���A�%*

logging/current_cost9�;O,�J+       ��K	q0��A�%*

logging/current_costk�;�(�+       ��K	�^��A�%*

logging/current_cost��;١�4+       ��K	����A�%*

logging/current_costi��;��_+       ��K	q���A�%*

logging/current_cost�߂;�fT>+       ��K	����A�%*

logging/current_costkނ;W���+       ��K	a��A�%*

logging/current_cost݂;.���+       ��K	�A��A�%*

logging/current_costM܂;�Z.+       ��K	�m��A�%*

logging/current_cost�ڂ;Pe�p+       ��K	.���A�%*

logging/current_costhڂ;�1+       ��K	����A�%*

logging/current_cost�؂;���+       ��K	���A�%*

logging/current_cost�ׂ;Z+G'+       ��K	5"��A�%*

logging/current_cost�ւ;�μ�+       ��K	O��A�&*

logging/current_costPՂ;����+       ��K	�}��A�&*

logging/current_costHԂ;����+       ��K	ӫ��A�&*

logging/current_cost-ӂ;*��+       ��K	~���A�&*

logging/current_cost�т;(�+       ��K	�	��A�&*

logging/current_costт;{k_+       ��K	�;��A�&*

logging/current_cost Ђ;��~|+       ��K	h��A�&*

logging/current_costrς;8��g+       ��K	����A�&*

logging/current_cost�͂;����+       ��K	���A�&*

logging/current_cost�̂;�K5+       ��K	���A�&*

logging/current_cost�˂; ��+       ��K	��A�&*

logging/current_cost{ʂ;��x9+       ��K	|J��A�&*

logging/current_cost�ɂ;	"�	+       ��K	�w��A�&*

logging/current_costBȂ;����+       ��K	����A�&*

logging/current_costǂ;���+       ��K	B���A�&*

logging/current_cost5Ƃ;*V-�+       ��K	^���A�&*

logging/current_cost�ł;d�#�+       ��K	�)���A�&*

logging/current_cost�Ă;��Q+       ��K	!X���A�&*

logging/current_cost"Â;��-+       ��K	Յ���A�&*

logging/current_cost ;>y�+       ��K	T����A�&*

logging/current_costu��;�Q��+       ��K	�����A�&*

logging/current_costп�;����+       ��K	T���A�&*

logging/current_costt��;� j+       ��K	�8���A�&*

logging/current_cost���;�30+       ��K	�f���A�&*

logging/current_costF��;�X�+       ��K		����A�&*

logging/current_cost���;[�*t+       ��K	F����A�&*

logging/current_cost���;��a=+       ��K	z����A�'*

logging/current_costA��;I�F+       ��K	�$���A�'*

logging/current_cost���;3:D\+       ��K	�U���A�'*

logging/current_cost��;i�m�+       ��K	����A�'*

logging/current_cost���;���+       ��K	Բ���A�'*

logging/current_costӴ�;� +       ��K	����A�'*

logging/current_costҳ�;���f+       ��K	����A�'*

logging/current_cost
��;�MZz+       ��K	�:���A�'*

logging/current_cost���;�X�!+       ��K	og���A�'*

logging/current_cost���;5�u�+       ��K	;����A�'*

logging/current_cost��;�n�+       ��K	�����A�'*

logging/current_cost��;���+       ��K	z����A�'*

logging/current_cost*��;�A�b+       ��K	�#���A�'*

logging/current_cost���;���#+       ��K	�R���A�'*

logging/current_cost۫�;�\�++       ��K	�����A�'*

logging/current_cost��;�5�+       ��K	�����A�'*

logging/current_cost���;Ẳ
+       ��K	7����A�'*

logging/current_costv��;���+       ��K	���A�'*

logging/current_costԧ�;��W�+       ��K	:D���A�'*

logging/current_cost��;N�#3+       ��K	
s���A�'*

logging/current_cost���;�*E+       ��K	�����A�'*

logging/current_cost���;�\�v+       ��K	�����A�'*

logging/current_cost좂;�h+       ��K	� ���A�'*

logging/current_cost���;`��+       ��K	.���A�'*

logging/current_cost���;��G�+       ��K	}Z���A�'*

logging/current_costL��;ә;+       ��K	9����A�(*

logging/current_costO��;���@+       ��K	�����A�(*

logging/current_cost-��;�s�+       ��K	1����A�(*

logging/current_costW��;_U�N+       ��K	����A�(*

logging/current_cost#��;��+       ��K	D���A�(*

logging/current_costΚ�;�r�+       ��K	v���A�(*

logging/current_costh��;6�+       ��K	A���A�(*

logging/current_costb��;�k��+       ��K		����A�(*

logging/current_cost���;�s!�+       ��K	 3���A�(*

logging/current_cost!��;4b/+       ��K	�s���A�(*

logging/current_cost���;k���+       ��K	h����A�(*

logging/current_cost-��;Ӷ�+       ��K	n����A�(*

logging/current_cost#��;�8,x+       ��K	����A�(*

logging/current_cost䑂;W�e+       ��K	7O���A�(*

logging/current_cost��;��ɱ+       ��K	����A�(*

logging/current_cost���;5��+       ��K	`����A�(*

logging/current_cost㎂;%���+       ��K	�����A�(*

logging/current_costˍ�;w2{�+       ��K	����A�(*

logging/current_cost���;7���+       ��K	qK���A�(*

logging/current_costዂ;��+       ��K	�~���A�(*

logging/current_costƊ�;��!+       ��K	����A�(*

logging/current_costǉ�;����+       ��K	p����A�(*

logging/current_cost߈�;��{�+       ��K	� ��A�(*

logging/current_cost���;��<%+       ��K	8 ��A�(*

logging/current_costO��;�?��+       ��K	�e ��A�(*

logging/current_cost݅�;�d2+       ��K	B� ��A�(*

logging/current_cost܄�;Z#�+       ��K	&� ��A�)*

logging/current_cost>��;��+       ��K	�� ��A�)*

logging/current_cost���;����+       ��K	�!��A�)*

logging/current_cost��;��m2+       ��K	�O��A�)*

logging/current_cost耂;% �+       ��K	c~��A�)*

logging/current_cost��;�9+       ��K	���A�)*

logging/current_costg~�;�k�+       ��K	M���A�)*

logging/current_cost~�;�_�`+       ��K	�
��A�)*

logging/current_costr|�;Ŕ�J+       ��K	6?��A�)*

logging/current_cost<{�;I�s�+       ��K	gl��A�)*

logging/current_cost{�;�X%+       ��K	p���A�)*

logging/current_cost�y�;1�i+       ��K	����A�)*

logging/current_costx�;��v<+       ��K	����A�)*

logging/current_cost�v�;}�+       ��K	 &��A�)*

logging/current_cost/v�;�i�+       ��K	fZ��A�)*

logging/current_cost*u�;1���+       ��K	����A�)*

logging/current_cost�s�;��q+       ��K	����A�)*

logging/current_cost�r�;��e+       ��K	O���A�)*

logging/current_cost�q�;hJ[+       ��K	��A�)*

logging/current_costMq�;��p�+       ��K	;M��A�)*

logging/current_costp�;��k�+       ��K	�z��A�)*

logging/current_costo�;�3�+       ��K	e���A�)*

logging/current_cost�m�;;��+       ��K	/���A�)*

logging/current_cost�l�;��1�+       ��K	E
��A�)*

logging/current_cost�k�;y[T�+       ��K	>9��A�)*

logging/current_cost�j�;���+       ��K	*g��A�)*

logging/current_cost�i�;����+       ��K	����A�**

logging/current_costi�;q�<+       ��K	����A�**

logging/current_cost:h�;ف�+       ��K	p���A�**

logging/current_cost�f�;�S+       ��K	/,��A�**

logging/current_cost�e�;��
+       ��K	L_��A�**

logging/current_cost�d�;w�+       ��K	���A�**

logging/current_cost�c�;�C��+       ��K	����A�**

logging/current_costc�;��A�+       ��K	^0��A�**

logging/current_cost�a�;O�m:+       ��K	}o��A�**

logging/current_cost�`�;a
[�+       ��K	0���A�**

logging/current_cost�_�;�e"+       ��K	����A�**

logging/current_cost�^�;��,6+       ��K	���A�**

logging/current_cost�]�;��w�+       ��K	"Z��A�**

logging/current_cost�\�;;d�I+       ��K	k���A�**

logging/current_cost=\�;e(�i+       ��K	W���A�**

logging/current_cost�Z�;q�D�+       ��K	�	��A�**

logging/current_costZ�;�>e�+       ��K	�9	��A�**

logging/current_cost�X�;���+       ��K	m	��A�**

logging/current_cost�W�;���7+       ��K	�	��A�**

logging/current_cost�V�;�ZW+       ��K	|�	��A�**

logging/current_cost,V�;���+       ��K	P
��A�**

logging/current_cost�T�;�A_<+       ��K	�3
��A�**

logging/current_cost�S�;�-+       ��K	�e
��A�**

logging/current_costS�;��f�+       ��K	��
��A�**

logging/current_costR�;�g<o+       ��K	&�
��A�**

logging/current_cost�P�;�jP+       ��K	��
��A�+*

logging/current_costP�;f/��+       ��K	���A�+*

logging/current_cost+O�;�ƈ�+       ��K	�J��A�+*

logging/current_cost�N�;�R5+       ��K	[z��A�+*

logging/current_cost-M�;�XU+       ��K	����A�+*

logging/current_costJL�;H��|+       ��K	����A�+*

logging/current_cost.K�;1�+       ��K	��A�+*

logging/current_cost]J�;
=�+       ��K	�5��A�+*

logging/current_cost�I�;Q3C�+       ��K	;h��A�+*

logging/current_cost+H�;L��+       ��K	���A�+*

logging/current_cost�G�;x�~�+       ��K	����A�+*

logging/current_cost�F�;޹�y+       ��K	����A�+*

logging/current_costwE�;4���+       ��K	$��A�+*

logging/current_cost�D�;[0'�+       ��K	D^��A�+*

logging/current_cost�C�;�D�+       ��K	����A�+*

logging/current_cost�B�;����+       ��K	{���A�+*

logging/current_cost�A�;�G��+       ��K	6���A�+*

logging/current_cost�@�;H�ڻ+       ��K	z��A�+*

logging/current_costK@�;���+       ��K	K��A�+*

logging/current_cost�>�;L��+       ��K	�}��A�+*

logging/current_cost'>�;�fP�+       ��K	����A�+*

logging/current_cost�=�;8�s�+       ��K	���A�+*

logging/current_cost-<�;���+       ��K	���A�+*

logging/current_cost<;�;�t}�+       ��K	�0��A�+*

logging/current_cost�:�;P}WU+       ��K	�^��A�+*

logging/current_costb9�;��+       ��K	���A�+*

logging/current_cost�8�;��A+       ��K	
���A�,*

logging/current_cost�7�;4b+       ��K	 ���A�,*

logging/current_cost�6�;��+       ��K	���A�,*

logging/current_cost�5�;v(��+       ��K	iF��A�,*

logging/current_cost�4�;�v�+       ��K	�z��A�,*

logging/current_cost�4�;��IK+       ��K	Ϫ��A�,*

logging/current_cost�3�;-���+       ��K	����A�,*

logging/current_cost�2�;]��'+       ��K		��A�,*

logging/current_costX1�;ɓ*[+       ��K	M6��A�,*

logging/current_cost}0�;�P!�+       ��K	*c��A�,*

logging/current_costL/�;y)@�+       ��K	[���A�,*

logging/current_costn.�;�Y{+       ��K	����A�,*

logging/current_cost�-�;�_2+       ��K	����A�,*

logging/current_cost�,�;��&+       ��K	���A�,*

logging/current_cost,�;z'e�+       ��K	�K��A�,*

logging/current_costr+�;h:��+       ��K	Q~��A�,*

logging/current_costp*�;w��c+       ��K	ܳ��A�,*

logging/current_cost,*�;��+       ��K	 ���A�,*

logging/current_costx(�;�h�+       ��K	C��A�,*

logging/current_costb'�;��e�+       ��K	^E��A�,*

logging/current_cost�&�;��+       ��K	Sw��A�,*

logging/current_cost&�;$Ş+       ��K	g���A�,*

logging/current_cost%�;5�+       ��K	����A�,*

logging/current_cost8$�;|��+       ��K	��A�,*

logging/current_cost=$�;ݍ<+       ��K	�C��A�,*

logging/current_cost #�;n �|+       ��K	�q��A�-*

logging/current_cost�!�;���Y+       ��K	���A�-*

logging/current_cost� �;��vC+       ��K	����A�-*

logging/current_cost��;����+       ��K	*��A�-*

logging/current_cost��;'$<�+       ��K	2��A�-*

logging/current_costo�;�+       ��K	�c��A�-*

logging/current_cost��;�,�+       ��K	+���A�-*

logging/current_cost��;닢�+       ��K	F���A�-*

logging/current_cost��;��0+       ��K	����A�-*

logging/current_cost��;g�I�+       ��K	I��A�-*

logging/current_cost��;/���+       ��K	�M��A�-*

logging/current_costI�;J<B)+       ��K	�{��A�-*

logging/current_cost��;y��+       ��K	S���A�-*

logging/current_cost��;�T��+       ��K	}���A�-*

logging/current_cost��;z,�H+       ��K	��A�-*

logging/current_cost�;䨇�+       ��K	b8��A�-*

logging/current_cost��;��w+       ��K	�e��A�-*

logging/current_cost��;s�;#+       ��K	?���A�-*

logging/current_cost��;���C+       ��K	����A�-*

logging/current_cost��;$��~+       ��K	���A�-*

logging/current_cost�;Lۯ.+       ��K	<"��A�-*

logging/current_cost��;	ъ�+       ��K	AO��A�-*

logging/current_costg�; �q+       ��K	�|��A�-*

logging/current_cost��;9�I+       ��K	����A�-*

logging/current_cost9�;m�<}+       ��K	x���A�-*

logging/current_cost��;���+       ��K	��A�-*

logging/current_costj�;�()�+       ��K	c4��A�.*

logging/current_costw�;ָ��+       ��K	&d��A�.*

logging/current_cost��;wo�c+       ��K	*���A�.*

logging/current_cost��;�7Y�+       ��K	x���A�.*

logging/current_cost9
�;ҥ�+       ��K	���A�.*

logging/current_cost�	�;N���+       ��K	���A�.*

logging/current_cost��;M <�+       ��K	VI��A�.*

logging/current_cost�;HX�+       ��K	\w��A�.*

logging/current_cost�;�	�+       ��K	���A�.*

logging/current_cost��;�:+       ��K	����A�.*

logging/current_cost%�;��s�+       ��K	r ��A�.*

logging/current_cost��;p�+       ��K	@/��A�.*

logging/current_costx�;bz�L+       ��K	x]��A�.*

logging/current_cost'�;�V�}+       ��K	���A�.*

logging/current_cost1�;:��+       ��K	/���A�.*

logging/current_cost\�;��+       ��K	���A�.*

logging/current_cost��;�y�+       ��K	���A�.*

logging/current_costl�;-�� +       ��K	=E��A�.*

logging/current_cost �;��D+       ��K	�r��A�.*

logging/current_cost���;gA2+       ��K	���A�.*

logging/current_cost��;Τ�n+       ��K	����A�.*

logging/current_cost���;>	A�+       ��K	����A�.*

logging/current_cost���;��+       ��K	�,��A�.*

logging/current_cost���;���+       ��K	C^��A�.*

logging/current_cost���;8��j+       ��K	���A�.*

logging/current_costY��;.J�v+       ��K	����A�.*

logging/current_costm��;l��h+       ��K	|���A�/*

logging/current_cost���;�ŶI+       ��K	l��A�/*

logging/current_costw��;)u��+       ��K	8E��A�/*

logging/current_cost}��;��]L+       ��K	�q��A�/*

logging/current_costx��;e�m�+       ��K	����A�/*

logging/current_costD��;U���+       ��K	E���A�/*

logging/current_costR��;KϞA+       ��K	���A�/*

logging/current_cost���;��/+       ��K	',��A�/*

logging/current_cost��;���f+       ��K	KY��A�/*

logging/current_cost��;�z+       ��K	����A�/*

logging/current_cost��;���+       ��K	����A�/*

logging/current_cost��;N�+       ��K	����A�/*

logging/current_cost��;D\��+       ��K	 ��A�/*

logging/current_cost�;�?�+       ��K	eG ��A�/*

logging/current_cost��;f�^{+       ��K	�w ��A�/*

logging/current_costf��;����+       ��K	C� ��A�/*

logging/current_cost��;�K��+       ��K	v� ��A�/*

logging/current_cost��;u_��+       ��K	�� ��A�/*

logging/current_costt�;YK��+       ��K	j-!��A�/*

logging/current_cost��;��w�+       ��K	�Y!��A�/*

logging/current_cost�;��B�+       ��K	e�!��A�/*

logging/current_cost��;�� �+       ��K	�!��A�/*

logging/current_costf�;F��Y+       ��K	J�!��A�/*

logging/current_cost��;v(r�+       ��K	m"��A�/*

logging/current_cost��;�c�+       ��K	tB"��A�/*

logging/current_costD�;��L+       ��K	9q"��A�0*

logging/current_costV�;pE��+       ��K	ݞ"��A�0*

logging/current_costr�;z�ϋ+       ��K	2�"��A�0*

logging/current_cost��;\+�q+       ��K	�"��A�0*

logging/current_cost��;�X�+       ��K	�$#��A�0*

logging/current_cost*�;i�F�+       ��K	V#��A�0*

logging/current_cost��;h�K�+       ��K	q�#��A�0*

logging/current_cost��;7���+       ��K	[�#��A�0*

logging/current_cost��;9+8�+       ��K	G�#��A�0*

logging/current_cost��;�]]+       ��K	s
$��A�0*

logging/current_cost��;��y+       ��K	�9$��A�0*

logging/current_costJ�;��+       ��K	�g$��A�0*

logging/current_cost]�;��y�+       ��K	
�$��A�0*

logging/current_cost+�;���+       ��K	+�$��A�0*

logging/current_cost��;1�զ+       ��K	`�$��A�0*

logging/current_cost���;V�{O+       ��K	]%��A�0*

logging/current_cost���;"���+       ��K	�H%��A�0*

logging/current_costg��;!��+       ��K	�u%��A�0*

logging/current_cost�߁;���+       ��K	z�%��A�0*

logging/current_cost9߁;HdQ+       ��K	Z�%��A�0*

logging/current_cost!߁;c:_�+       ��K	m�%��A�0*

logging/current_cost�݁;��2+       ��K	�+&��A�0*

logging/current_cost�܁;��-�+       ��K	HZ&��A�0*

logging/current_costo܁;$�_�+       ��K	:�&��A�0*

logging/current_cost-܁;��I+       ��K	��&��A�0*

logging/current_cost�ہ;�`��+       ��K	��&��A�0*

logging/current_costHہ;0��I+       ��K	�'��A�1*

logging/current_cost�ځ;��C�+       ��K	mD'��A�1*

logging/current_cost�ف;�v��+       ��K	�p'��A�1*

logging/current_cost`ف;�a�+       ��K	��'��A�1*

logging/current_cost�؁;u��+       ��K	��'��A�1*

logging/current_cost�؁;Ҫ_+       ��K	��'��A�1*

logging/current_cost�ׁ;?�8g+       ��K	".(��A�1*

logging/current_cost3ׁ;��f#+       ��K	�\(��A�1*

logging/current_costKց;��+       ��K	ʋ(��A�1*

logging/current_cost�Ձ;�<f�+       ��K	ʹ(��A�1*

logging/current_cost�Ձ;�wT+       ��K	;�(��A�1*

logging/current_cost�ԁ;Fz+       ��K	�)��A�1*

logging/current_costtԁ;l�i+       ��K	�?)��A�1*

logging/current_cost�Ӂ;S��+       ��K	zk)��A�1*

logging/current_costӁ;ʝ)+       ��K	;�)��A�1*

logging/current_costӁ;���+       ��K	�)��A�1*

logging/current_costEҁ;L�.+       ��K	 �)��A�1*

logging/current_cost�ҁ;��
+       ��K	+*��A�1*

logging/current_cost�с;c���+       ��K	�^*��A�1*

logging/current_cost�с;O��+       ��K	��*��A�1*

logging/current_cost с;<풽+       ��K	��*��A�1*

logging/current_costЁ;�iZ�+       ��K	>�*��A�1*

logging/current_costρ;�$+       ��K	�++��A�1*

logging/current_cost�΁;���+       ��K	Ac+��A�1*

logging/current_cost�΁;���+       ��K	Ȝ+��A�1*

logging/current_cost�́;�v�+       ��K	�+��A�2*

logging/current_cost́;����+       ��K	�,��A�2*

logging/current_cost�́;s��7+       ��K	�A,��A�2*

logging/current_cost=́;�3P�+       ��K	-q,��A�2*

logging/current_cost�ˁ;���+       ��K	-�,��A�2*

logging/current_cost{ˁ;�'��+       ��K	��,��A�2*

logging/current_costzˁ;pkY+       ��K	-��A�2*

logging/current_costRʁ;�;�+       ��K	�<-��A�2*

logging/current_cost�ʁ;�Ú�+       ��K	Ss-��A�2*

logging/current_costʁ;�	�+       ��K	%�-��A�2*

logging/current_cost�ȁ;}]��+       ��K	��-��A�2*

logging/current_costȁ;_]bZ+       ��K	�.��A�2*

logging/current_cost�ǁ;��+       ��K	�C.��A�2*

logging/current_cost�ǁ;A;+�+       ��K	�v.��A�2*

logging/current_costǁ;3���+       ��K	7�.��A�2*

logging/current_cost�Ɓ;�Ba�+       ��K	��.��A�2*

logging/current_cost�Ɓ;�4��+       ��K	�/��A�2*

logging/current_cost�Ł;���+       ��K	p>/��A�2*

logging/current_cost�Ł;�9u+       ��K	t/��A�2*

logging/current_cost�ā;�ʇ+       ��K	ܣ/��A�2*

logging/current_cost>ā;0��+       ��K	��/��A�2*

logging/current_cost�Á; k�j+       ��K	��/��A�2*

logging/current_cost�Á;
Y�:+       ��K	J/0��A�2*

logging/current_costÁ;�`��+       ��K	;^0��A�2*

logging/current_cost�;�T +       ��K	^�0��A�2*

logging/current_cost�;�Bh+       ��K	3�0��A�2*

logging/current_cost���;:�i+       ��K	 �0��A�3*

logging/current_cost(��;�ۯZ+       ��K	L1��A�3*

logging/current_costi��;*��+       ��K	�B1��A�3*

logging/current_cost���;_��+       ��K	?p1��A�3*

logging/current_cost���;a>y<+       ��K	��1��A�3*

logging/current_cost-��;69�+       ��K	_�1��A�3*

logging/current_cost���;��(p+       ��K	��1��A�3*

logging/current_cost龁;ܐyp+       ��K	�&2��A�3*

logging/current_cost���;�J5+       ��K	�U2��A�3*

logging/current_cost���;
��	+       ��K	؁2��A�3*

logging/current_costý�;3�`C+       ��K	��2��A�3*

logging/current_cost���;�U{+       ��K	�2��A�3*

logging/current_cost���;?N�q+       ��K	�3��A�3*

logging/current_cost ��;Y��+       ��K	�53��A�3*

logging/current_costѻ�;�P=:+       ��K	1c3��A�3*

logging/current_costs��;�o+       ��K	7�3��A�3*

logging/current_cost̻�;(R�;+       ��K	��3��A�3*

logging/current_cost\��;�Fw�+       ��K	L�3��A�3*

logging/current_costĺ�;�\ݴ+       ��K	V4��A�3*

logging/current_cost���;���6+       ��K	�F4��A�3*

logging/current_cost)��;/9�+       ��K	�r4��A�3*

logging/current_costƸ�;�y��+       ��K	H�4��A�3*

logging/current_cost���;�kw+       ��K	��4��A�3*

logging/current_cost���;�{+       ��K	��4��A�3*

logging/current_cost���;�f��+       ��K	�)5��A�3*

logging/current_cost&��;M~+       ��K	4V5��A�3*

logging/current_cost���;��+       ��K	?�5��A�4*

logging/current_costd��;�W�{+       ��K	v�5��A�4*

logging/current_cost絁;�=��+       ��K		�5��A�4*

logging/current_cost���;�$��+       ��K		6��A�4*

logging/current_cost7��;���+       ��K	
56��A�4*

logging/current_cost���;h�~+       ��K	�a6��A�4*

logging/current_cost���;�*Ū+       ��K	i�6��A�4*

logging/current_cost~��;k*+       ��K	��6��A�4*

logging/current_cost���;�U\�+       ��K	��6��A�4*

logging/current_costn��;�++       ��K	�7��A�4*

logging/current_cost0��;R��+       ��K	D7��A�4*

logging/current_cost���;m�+       ��K	�q7��A�4*

logging/current_cost��;��t+       ��K	��7��A�4*

logging/current_costT��;��Ax+       ��K	��7��A�4*

logging/current_costɲ�;�dS+       ��K	�8��A�4*

logging/current_cost6��;�J��+       ��K	�28��A�4*

logging/current_cost��;ϓ��+       ��K	�a8��A�4*

logging/current_costA��;[�MR+       ��K	 �8��A�4*

logging/current_cost,��;97y�+       ��K	S�8��A�4*

logging/current_cost��;�«I+       ��K	�8��A�4*

logging/current_cost���;O+��+       ��K	D"9��A�4*

logging/current_cost[��;�H�J+       ��K	�S9��A�4*

logging/current_cost1��;F�e�+       ��K	P�9��A�4*

logging/current_cost���;��]7+       ��K	Z�9��A�4*

logging/current_costb��;���W+       ��K	��9��A�4*

logging/current_costw��;��I�+       ��K	�:��A�5*

logging/current_costj��;�;��+       ��K	�>:��A�5*

logging/current_cost2��;c�B	+       ��K	�k:��A�5*

logging/current_cost���;��9+       ��K	2�:��A�5*

logging/current_cost[��;"�j�+       ��K	_�:��A�5*

logging/current_cost ��;�`�+       ��K	��:��A�5*

logging/current_cost櫁;B�;+       ��K	�(;��A�5*

logging/current_costߪ�;�J��+       ��K	P\;��A�5*

logging/current_costҪ�;�:��+       ��K	�;��A�5*

logging/current_costn��;��K+       ��K	��;��A�5*

logging/current_costZ��;|v�+       ��K	a<��A�5*

logging/current_cost���;]�0>+       ��K	jO<��A�5*

logging/current_costʨ�;��0�+       ��K	<�<��A�5*

logging/current_costa��;&�(�+       ��K	V�<��A�5*

logging/current_cost���;�Ī�+       ��K	�=��A�5*

logging/current_cost⨁;F˕+       ��K	�E=��A�5*

logging/current_cost㧁;���+       ��K	_�=��A�5*

logging/current_cost��;ҽ�J+       ��K	��=��A�5*

logging/current_costj��;�J�A+       ��K	�>��A�5*

logging/current_costǦ�;eN@�+       ��K	�<>��A�5*

logging/current_cost���;�&P+       ��K	}q>��A�5*

logging/current_cost���;����+       ��K	ҧ>��A�5*

logging/current_cost���;���+       ��K	��>��A�5*

logging/current_cost���;��+       ��K	~?��A�5*

logging/current_cost��;%��+       ��K	aH?��A�5*

logging/current_cost���;h���+       ��K	b�?��A�5*

logging/current_cost更;[EG�+       ��K	C�?��A�6*

logging/current_costH��;�% +       ��K	#�?��A�6*

logging/current_cost���;X݆d+       ��K	@��A�6*

logging/current_cost�;��<e+       ��K	�T@��A�6*

logging/current_cost͢�;�%�+       ��K	H�@��A�6*

logging/current_cost^��;��<�+       ��K	��@��A�6*

logging/current_cost٢�;(�I+       ��K	A��A�6*

logging/current_costT��;���+       ��K	%FA��A�6*

logging/current_cost(��;���+       ��K	�~A��A�6*

logging/current_cost[��;��+       ��K	�A��A�6*

logging/current_cost렁;�fr+       ��K	��A��A�6*

logging/current_costˠ�;�9+       ��K	H!B��A�6*

logging/current_cost��;c��+       ��K	.^B��A�6*

logging/current_cost&��;�p2+       ��K	p�B��A�6*

logging/current_cost,��;��+       ��K	��B��A�6*

logging/current_cost���;���+       ��K	a�B��A�6*

logging/current_costn��;Ťu+       ��K	�<C��A�6*

logging/current_costP��;��?+       ��K	�uC��A�6*

logging/current_cost���;���-+       ��K	ڪC��A�6*

logging/current_cost���;``��+       ��K	c�C��A�6*

logging/current_costm��;�|2�+       ��K	�
D��A�6*

logging/current_costu��;�Eb+       ��K	�BD��A�6*

logging/current_cost���;�χ+       ��K	�tD��A�6*

logging/current_cost��;�/�]+       ��K	M�D��A�6*

logging/current_cost���;J��+       ��K	{�D��A�6*

logging/current_cost	��;ɷ��+       ��K	nE��A�7*

logging/current_costȜ�;�C�+       ��K	(3E��A�7*

logging/current_cost훁;K�/�+       ��K	�dE��A�7*

logging/current_costh��;V�0+       ��K	ՓE��A�7*

logging/current_cost��;Ԍ��+       ��K	;�E��A�7*

logging/current_cost���;2��+       ��K	��E��A�7*

logging/current_cost4��;0T�R+       ��K	�,F��A�7*

logging/current_costŘ�;�,*>+       ��K	ceF��A�7*

logging/current_cost#��;o3+       ��K	�F��A�7*

logging/current_cost���;�n�/+       ��K	��F��A�7*

logging/current_cost���;
I|�+       ��K	��F��A�7*

logging/current_cost��;��+       ��K	Z.G��A�7*

logging/current_costߖ�;#���+       ��K	�aG��A�7*

logging/current_cost6��;0+wo+       ��K	6�G��A�7*

logging/current_costy��;8��++       ��K	q�G��A�7*

logging/current_costR��;W���+       ��K	i�G��A�7*

logging/current_cost━;@�}�+       ��K	�"H��A�7*

logging/current_cost���;F�Yp+       ��K	|SH��A�7*

logging/current_cost��;��]�+       ��K	��H��A�7*

logging/current_cost)��;���+       ��K	ǴH��A�7*

logging/current_cost�;:�!�+       ��K	��H��A�7*

logging/current_cost���;�G+       ��K	�I��A�7*

logging/current_cost���;�eX+       ��K	0BI��A�7*

logging/current_cost���;���+       ��K	�pI��A�7*

logging/current_costǒ�;g|��+       ��K	�I��A�7*

logging/current_costs��;�ޫ%+       ��K	%�I��A�7*

logging/current_cost1��;J0+       ��K	��I��A�8*

logging/current_cost���;�&Q+       ��K	�1J��A�8*

logging/current_cost��;�XR+       ��K	�bJ��A�8*

logging/current_cost��;9(+       ��K	�J��A�8*

logging/current_cost���;u�X+       ��K	��J��A�8*

logging/current_cost���;M�8�+       ��K	��J��A�8*

logging/current_cost뎁;mfc�+       ��K	�K��A�8*

logging/current_costL��;w�l+       ��K	�PK��A�8*

logging/current_cost���;	e'+       ��K	�K��A�8*

logging/current_costǎ�;�UT+       ��K	ګK��A�8*

logging/current_cost��;���9+       ��K	�K��A�8*

logging/current_costʍ�;~��/+       ��K	sL��A�8*

logging/current_cost���;��Jr+       ��K	V3L��A�8*

logging/current_cost���;)�Zn+       ��K	qfL��A�8*

logging/current_cost;Jk�+       ��K	~�L��A�8*

logging/current_cost���;�n��+       ��K	_�L��A�8*

logging/current_cost���;�G�+       ��K	
�L��A�8*

logging/current_costd��;{��	+       ��K	�$M��A�8*

logging/current_cost���;h`-�+       ��K	fXM��A�8*

logging/current_costދ�;a�+       ��K	�M��A�8*

logging/current_coste��;'+       ��K	�M��A�8*

logging/current_cost���;��٧+       ��K	s�M��A�8*

logging/current_cost���;�\��+       ��K	�N��A�8*

logging/current_cost��;�C@�+       ��K	�@N��A�8*

logging/current_cost���;�,��+       ��K	�tN��A�8*

logging/current_costˉ�;��n+       ��K	�N��A�8*

logging/current_costg��;%��l+       ��K	��N��A�9*

logging/current_cost��;��Q�+       ��K	9�N��A�9*

logging/current_cost<��;Y+       ��K	�*O��A�9*

logging/current_costȉ�;�L��+       ��K	]YO��A�9*

logging/current_cost5��;!�~�+       ��K	"�O��A�9*

logging/current_costm��;�,�+       ��K	F�O��A�9*

logging/current_costꇁ;��&+       ��K	4�O��A�9*

logging/current_costՇ�;>yM+       ��K	3P��A�9*

logging/current_cost·�;�s-�+       ��K	lHP��A�9*

logging/current_cost��;�g�+       ��K	�uP��A�9*

logging/current_cost���;���+       ��K	ըP��A�9*

logging/current_cost���;FE0+       ��K	��P��A�9*

logging/current_cost5��;�4��+       ��K	XQ��A�9*

logging/current_cost���;��k�+       ��K	�4Q��A�9*

logging/current_cost΅�;Υ�'+       ��K	�eQ��A�9*

logging/current_costu��;����+       ��K	��Q��A�9*

logging/current_cost脁;��+       ��K	$�Q��A�9*

logging/current_cost���;p�U+       ��K	:�Q��A�9*

logging/current_costӄ�;���u+       ��K	�#R��A�9*

logging/current_cost~��;���n+       ��K	�VR��A�9*

logging/current_cost�;Q<��+       ��K	��R��A�9*

logging/current_cost���;P�+       ��K	n�R��A�9*

logging/current_costx��;�u�+       ��K	t�R��A�9*

logging/current_cost���;=�=+       ��K	�S��A�9*

logging/current_cost-��;}Fܩ+       ��K	�AS��A�9*

logging/current_cost2��;��J+       ��K	�oS��A�:*

logging/current_costǂ�;�5�+       ��K	�S��A�:*

logging/current_cost��;f7��+       ��K	��S��A�:*

logging/current_cost���;a���+       ��K	�T��A�:*

logging/current_cost���;��~S+       ��K	�5T��A�:*

logging/current_costt��;���=+       ��K	�dT��A�:*

logging/current_cost?�;��D+       ��K	��T��A�:*

logging/current_cost;�;X��_+       ��K	��T��A�:*

logging/current_cost��;A%�b+       ��K	��T��A�:*

logging/current_cost�;��+       ��K	QU��A�:*

logging/current_cost�~�;䭘�+       ��K	�HU��A�:*

logging/current_cost~�;�&F+       ��K	�uU��A�:*

logging/current_cost�}�;�)9+       ��K	ǡU��A�:*

logging/current_cost}�;����+       ��K	��U��A�:*

logging/current_costG|�;���+       ��K	F�U��A�:*

logging/current_cost]|�;:��?+       ��K	�/V��A�:*

logging/current_cost|�;�`9l+       ��K	�\V��A�:*

logging/current_cost�|�;K1*+       ��K	B�V��A�:*

logging/current_cost7|�; ��+       ��K	��V��A�:*

logging/current_cost�{�;C��?+       ��K	��V��A�:*

logging/current_cost{�;��iX+       ��K	�W��A�:*

logging/current_costo{�;<�_�+       ��K	TBW��A�:*

logging/current_cost�z�;���+       ��K	�uW��A�:*

logging/current_cost�y�;G��+       ��K	y�W��A�:*

logging/current_cost�z�;]XkA+       ��K	��W��A�:*

logging/current_cost{�;�Ը�+       ��K	� X��A�:*

logging/current_cost�{�;1�|�+       ��K	�/X��A�;*

logging/current_cost0{�;�@/�+       ��K	n^X��A�;*

logging/current_costz�;�)N+       ��K	��X��A�;*

logging/current_cost2y�;��>�+       ��K	��X��A�;*

logging/current_cost�w�;go��+       ��K	��X��A�;*

logging/current_cost�w�;��\+       ��K	Y��A�;*

logging/current_costQw�;��R8+       ��K	HY��A�;*

logging/current_costPv�;G�>�+       ��K	�wY��A�;*

logging/current_cost�u�;�g�+       ��K	�Y��A�;*

logging/current_cost�v�;2/e+       ��K	z�Y��A�;*

logging/current_cost�u�;pvb+       ��K	�Z��A�;*

logging/current_costu�;�+�+       ��K	2/Z��A�;*

logging/current_cost!t�;�`��+       ��K	+\Z��A�;*

logging/current_cost�t�;��6"+       ��K	P�Z��A�;*

logging/current_cost�t�;�Tg�+       ��K	f�Z��A�;*

logging/current_cost�s�;hQ� +       ��K	��Z��A�;*

logging/current_cost�t�;�U+       ��K	�[��A�;*

logging/current_cost`t�;��T+       ��K	@[��A�;*

logging/current_cost�s�;��%�+       ��K	Rl[��A�;*

logging/current_cost�r�;=)��+       ��K	z�[��A�;*

logging/current_cost�r�;bSK+       ��K	��[��A�;*

logging/current_costr�;~dz+       ��K	�[��A�;*

logging/current_cost2r�;Fz�+       ��K	'%\��A�;*

logging/current_cost�q�;�X�+       ��K	�Q\��A�;*

logging/current_cost�q�;��@J+       ��K	�~\��A�;*

logging/current_cost�p�;C��&+       ��K	r�\��A�<*

logging/current_cost�p�;�=�K+       ��K	��\��A�<*

logging/current_cost�q�;�+�+       ��K	K]��A�<*

logging/current_costq�;��F+       ��K	�4]��A�<*

logging/current_cost�q�;��??+       ��K	�a]��A�<*

logging/current_costSp�;�n]+       ��K	��]��A�<*

logging/current_costio�;x#�+       ��K	H�]��A�<*

logging/current_cost�o�;^��+       ��K	��]��A�<*

logging/current_costo�;E��+       ��K	�^��A�<*

logging/current_cost�o�;�&"+       ��K	!@^��A�<*

logging/current_costPn�;��_+       ��K	�q^��A�<*

logging/current_cost�m�;���+       ��K	 �^��A�<*

logging/current_cost6n�;!znc+       ��K	��^��A�<*

logging/current_cost1o�;��+       ��K	� _��A�<*

logging/current_cost�n�;��ْ+       ��K	�-_��A�<*

logging/current_cost9m�;��li+       ��K	�[_��A�<*

logging/current_costZm�;�t@�+       ��K	k�_��A�<*

logging/current_cost5m�;��b+       ��K	�_��A�<*

logging/current_cost�l�;� +       ��K	��_��A�<*

logging/current_cost�l�;��Fb+       ��K	�`��A�<*

logging/current_costl�;�?N�+       ��K	�B`��A�<*

logging/current_costCl�;e��C+       ��K	p`��A�<*

logging/current_cost�k�;
K�E+       ��K	��`��A�<*

logging/current_cost�k�;@弲+       ��K	��`��A�<*

logging/current_cost�j�;���:+       ��K	�`��A�<*

logging/current_costk�;�b��+       ��K	C-a��A�<*

logging/current_cost�k�;�(�0+       ��K	�\a��A�=*

logging/current_costk�;!�l+       ��K	�a��A�=*

logging/current_cost�j�;C,�+       ��K	�a��A�=*

logging/current_cost�i�;�E+       ��K	 �a��A�=*

logging/current_cost)i�;ؽ�+       ��K	b��A�=*

logging/current_cost�i�;z1�+       ��K	�Ib��A�=*

logging/current_cost�j�;kpG�+       ��K	�vb��A�=*

logging/current_cost�h�;#/9+       ��K	��b��A�=*

logging/current_costih�;�w�+       ��K	��b��A�=*

logging/current_cost�h�;)=��+       ��K	% c��A�=*

logging/current_cost1i�;��m�+       ��K	u-c��A�=*

logging/current_cost�i�;��+       ��K	uZc��A�=*

logging/current_cost�h�;�W+       ��K	��c��A�=*

logging/current_cost�f�;ʓ�?+       ��K	��c��A�=*

logging/current_cost]h�;I�'+       ��K	R�c��A�=*

logging/current_costTh�;9�+       ��K	�d��A�=*

logging/current_cost�g�;�
��+       ��K	�Jd��A�=*

logging/current_cost�f�;P��+       ��K	�wd��A�=*

logging/current_cost�f�;H1�R+       ��K	,�d��A�=*

logging/current_costWf�;�hn+       ��K	7�d��A�=*

logging/current_cost&f�; n+       ��K	8 e��A�=*

logging/current_cost`f�;�"ڔ+       ��K	c-e��A�=*

logging/current_cost�d�;o^�)+       ��K	�Ze��A�=*

logging/current_cost>e�;�}�N+       ��K	4�e��A�=*

logging/current_costfe�;%4�v+       ��K	��e��A�=*

logging/current_cost�d�;Q��+       ��K	(�e��A�=*

logging/current_costQe�;����+       ��K	Bf��A�>*

logging/current_costd�;j�c+       ��K	:?f��A�>*

logging/current_cost�c�;a��'+       ��K	�mf��A�>*

logging/current_costd�;ZH~+       ��K	��f��A�>*

logging/current_cost�d�;���/+       ��K	a�f��A�>*

logging/current_cost;c�;CQ�+       ��K	��f��A�>*

logging/current_costAc�;��ͷ+       ��K	&+g��A�>*

logging/current_costc�;G��+       ��K	�Xg��A�>*

logging/current_costZc�;ɋ+       ��K	�g��A�>*

logging/current_costc�;�X3)+       ��K	��g��A�>*

logging/current_costb�;���+       ��K	V�g��A�>*

logging/current_cost�a�;{��+       ��K	8h��A�>*

logging/current_costb�;��"r+       ��K	�>h��A�>*

logging/current_costFc�;�zG�+       ��K	lh��A�>*

logging/current_cost�c�;6�WA+       ��K	�h��A�>*

logging/current_cost�a�;�+       ��K	�h��A�>*

logging/current_costIa�;���I+       ��K	u�h��A�>*

logging/current_cost�`�;޵��+       ��K	Q!i��A�>*

logging/current_cost(a�;��m�+       ��K	*Mi��A�>*

logging/current_cost�a�;U��V+       ��K	�zi��A�>*

logging/current_cost�_�;�_�r+       ��K	��i��A�>*

logging/current_costj`�;��3+       ��K	\�i��A�>*

logging/current_cost!a�;Т'+       ��K	�j��A�>*

logging/current_cost�`�;ښ�+       ��K	�:j��A�>*

logging/current_cost�_�;
���+       ��K	�fj��A�>*

logging/current_cost9_�;G��+       ��K	��j��A�?*

logging/current_cost^�;	F:#+       ��K	?�j��A�?*

logging/current_costU^�;9�8�+       ��K	��j��A�?*

logging/current_cost�^�;�TQ�+       ��K	�k��A�?*

logging/current_cost�^�;���+       ��K	�Gk��A�?*

logging/current_cost�]�;؅�V+       ��K	�vk��A�?*

logging/current_cost�]�;��+       ��K	�k��A�?*

logging/current_costr_�;i{��+       ��K	�k��A�?*

logging/current_cost3^�;+r�+       ��K	kl��A�?*

logging/current_cost^�;P{�3+       ��K	�=l��A�?*

logging/current_cost~\�;ȋ�+       ��K	<ml��A�?*

logging/current_costq\�;��+       ��K	+�l��A�?*

logging/current_cost\�;���N+       ��K	N�l��A�?*

logging/current_cost�\�;4���+       ��K	Y�l��A�?*

logging/current_cost^]�;{Y�d+       ��K	�'m��A�?*

logging/current_cost#\�;޾�4+       ��K	WXm��A�?*

logging/current_costp[�;)6;+       ��K	Q�m��A�?*

logging/current_cost1\�;�e@+       ��K	��m��A�?*

logging/current_costY\�;P�ˏ+       ��K	0�m��A�?*

logging/current_cost\�;�=Ó+       ��K	�n��A�?*

logging/current_costI[�;����+       ��K	�An��A�?*

logging/current_cost?Z�;��|+       ��K	npn��A�?*

logging/current_cost#[�;r?�j+       ��K	��n��A�?*

logging/current_cost�Z�;@*b�+       ��K	��n��A�?*

logging/current_cost�Z�;G�+       ��K	9o��A�?*

logging/current_cost?Z�;p�d+       ��K	�/o��A�?*

logging/current_cost�Y�;f��+       ��K	]o��A�@*

logging/current_costTY�;���+       ��K	'�o��A�@*

logging/current_cost�Y�;�F_s+       ��K	Ȼo��A�@*

logging/current_cost�Y�;'ƯG+       ��K	��o��A�@*

logging/current_cost�X�;gV'-+       ��K	p��A�@*

logging/current_cost�Y�;Y�*�+       ��K	%Jp��A�@*

logging/current_cost�X�;��UI+       ��K	�zp��A�@*

logging/current_costX�;,��+       ��K	(�p��A�@*

logging/current_cost�X�;!F��+       ��K	-�p��A�@*

logging/current_costX�;�b�%+       ��K	�q��A�@*

logging/current_cost8W�;���+       ��K	t3q��A�@*

logging/current_cost�W�;�$�+       ��K	�aq��A�@*

logging/current_costX�;b�̋+       ��K	:�q��A�@*

logging/current_cost?X�;K8��+       ��K	f�q��A�@*

logging/current_costW�;Q��H+       ��K	��q��A�@*

logging/current_costJV�;�M�+       ��K	Xr��A�@*

logging/current_cost�X�;��-�+       ��K	AIr��A�@*

logging/current_costJZ�;;��\+       ��K	�ur��A�@*

logging/current_cost W�;?rQ+       ��K	��r��A�@*

logging/current_costV�;!M��+       ��K	�r��A�@*

logging/current_cost@U�;�)�+       ��K	��r��A�@*

logging/current_cost�U�;5Z�4+       ��K	�%s��A�@*

logging/current_cost"U�;����+       ��K	�Rs��A�@*

logging/current_cost�U�;�k��+       ��K	T�s��A�@*

logging/current_cost�T�;��?�+       ��K	1�s��A�@*

logging/current_cost�S�;Z^T�+       ��K	��s��A�A*

logging/current_cost�T�;�L��+       ��K	Pt��A�A*

logging/current_cost4U�;}Uw4+       ��K	�9t��A�A*

logging/current_costT�;��+       ��K	Ght��A�A*

logging/current_cost�U�;���-+       ��K	i�t��A�A*

logging/current_costW�;SN)�+       ��K	��t��A�A*

logging/current_cost�V�;��b+       ��K		�t��A�A*

logging/current_cost�T�;[�+       ��K	Mu��A�A*

logging/current_cost�S�;����+       ��K	/Lu��A�A*

logging/current_cost�M�;Q_�+       ��K	�zu��A�A*

logging/current_cost�G�;$�y�+       ��K	x�u��A�A*

logging/current_cost�?�;ʝ+       ��K	2�u��A�A*

logging/current_costj5�;��.+       ��K	Qv��A�A*

logging/current_cost��;�)�+       ��K	�4v��A�A*

logging/current_cost�;��c0+       ��K	Pbv��A�A*

logging/current_cost9�;Mb+       ��K	��v��A�A*

logging/current_cost��;�[�8+       ��K	r�v��A�A*

logging/current_cost��;���+       ��K	J�v��A�A*

logging/current_cost�;�C��+       ��K	)w��A�A*

logging/current_cost��;�g��+       ��K	9Gw��A�A*

logging/current_cost�;�T�+       ��K	btw��A�A*

logging/current_cost�܀;�Ə�+       ��K	�w��A�A*

logging/current_cost�ۀ;��>�+       ��K	��w��A�A*

logging/current_cost[ڀ;���(+       ��K	��w��A�A*

logging/current_cost.؀;���O+       ��K	D)x��A�A*

logging/current_cost�؀;2F�+       ��K	<Vx��A�A*

logging/current_cost�Հ;���+       ��K	 �x��A�B*

logging/current_cost�؀;z�.�+       ��K	�x��A�B*

logging/current_cost�Ԁ;H��+       ��K	{�x��A�B*

logging/current_costҀ;T�	+       ��K	:y��A�B*

logging/current_costIӀ;�9(i+       ��K	fFy��A�B*

logging/current_cost�π;��e+       ��K	ty��A�B*

logging/current_costڀ;�1x�+       ��K	i�y��A�B*

logging/current_cost%Ѐ;.��+       ��K	��y��A�B*

logging/current_costSӀ;�b�+       ��K	� z��A�B*

logging/current_cost9р;�؝3+       ��K	.z��A�B*

logging/current_cost΀;wc�\+       ��K	�`z��A�B*

logging/current_costỳ;�bz+       ��K	^�z��A�B*

logging/current_cost�̀;j��E+       ��K	�z��A�B*

logging/current_cost�̀;�Mx`+       ��K	��z��A�B*

logging/current_cost ˀ;�wd+       ��K	Z{��A�B*

logging/current_cost%ǀ;��"D+       ��K	�J{��A�B*

logging/current_cost�ǀ;�|��+       ��K	��{��A�B*

logging/current_cost�ŀ;P�`(+       ��K	��{��A�B*

logging/current_cost�̀;k�W+       ��K	��{��A�B*

logging/current_cost�Ā;!=�+       ��K	�-|��A�B*

logging/current_cost���;�D��+       ��K	�Z|��A�B*

logging/current_cost���;H�0+       ��K	�|��A�B*

logging/current_cost�;�5�+       ��K	�|��A�B*

logging/current_costɽ�;����+       ��K	��|��A�B*

logging/current_costJ��;q(#+       ��K	�}��A�B*

logging/current_cost���;���+       ��K	�J}��A�B*

logging/current_cost��;~,�+       ��K	%z}��A�C*

logging/current_cost���;"���+       ��K	��}��A�C*

logging/current_costx��;��y^+       ��K	K�}��A�C*

logging/current_cost��;���+       ��K	p~��A�C*

logging/current_cost���;2�-W+       ��K	�2~��A�C*

logging/current_cost���;�bN9+       ��K	�`~��A�C*

logging/current_cost��;���+       ��K	׎~��A�C*

logging/current_costɷ�;�R�t+       ��K	'�~��A�C*

logging/current_cost��;��ŋ+       ��K	a�~��A�C*

logging/current_cost���;�+.+       ��K	���A�C*

logging/current_costF��;h�+       ��K	kJ��A�C*

logging/current_cost���;U��M+       ��K	<y��A�C*

logging/current_cost;��;QQ`�+       ��K	:���A�C*

logging/current_cost°�;��g=+       ��K	9���A�C*

logging/current_cost��;:��+       ��K	����A�C*

logging/current_cost��;	�-#+       ��K	61���A�C*

logging/current_costѩ�;�e�9+       ��K	-_���A�C*

logging/current_costs��;Cc~)+       ��K	����A�C*

logging/current_cost)��;�IQ+       ��K	�����A�C*

logging/current_cost/��;K��+       ��K	�뀔�A�C*

logging/current_cost��;*R#+       ��K	���A�C*

logging/current_costɧ�;O��d+       ��K	�K���A�C*

logging/current_cost=��;R��'+       ��K	+y���A�C*

logging/current_cost4��;�u��+       ��K	�����A�C*

logging/current_cost���;D�O+       ��K	Uׁ��A�C*

logging/current_cost���;6,+       ��K	���A�D*

logging/current_cost��;��+       ��K	4���A�D*

logging/current_cost/��;B���+       ��K	~a���A�D*

logging/current_costԡ�;V��+       ��K	�����A�D*

logging/current_cost֠�;S��+       ��K	-����A�D*

logging/current_cost���;|�+       ��K	^�A�D*

logging/current_cost���;>��+       ��K	����A�D*

logging/current_cost���;�n+       ��K	�I���A�D*

logging/current_costv��;�RS+       ��K	�y���A�D*

logging/current_cost���;�f�o+       ��K	.����A�D*

logging/current_coste��;O��+       ��K	׃��A�D*

logging/current_cost��;�&.=+       ��K	1���A�D*

logging/current_costr��;ٝ��+       ��K	�1���A�D*

logging/current_cost˕�;.��+       ��K	7i���A�D*

logging/current_cost���;��kK+       ��K	�����A�D*

logging/current_costǘ�;[W�4+       ��K	�����A�D*

logging/current_cost���;W��q+       ��K	��A�D*

logging/current_costh��;15R+       ��K	$���A�D*

logging/current_costӗ�;\�&+       ��K	K���A�D*

logging/current_cost��;��]O+       ��K	�w���A�D*

logging/current_costk��;���%+       ��K	_����A�D*

logging/current_cost⍀;ybK+       ��K	�܅��A�D*

logging/current_cost���;�o(^+       ��K	����A�D*

logging/current_costV��;D��+       ��K	D���A�D*

logging/current_cost���;�ͨ�+       ��K	�u���A�D*

logging/current_cost葀;��U_+       ��K	�����A�D*

logging/current_costq��;�'��+       ��K	�Ն��A�E*

logging/current_costˋ�;���+       ��K	(���A�E*

logging/current_cost��;(o��+       ��K	�6���A�E*

logging/current_costX��;{�^�+       ��K	'c���A�E*

logging/current_costV��;P� �+       ��K	�����A�E*

logging/current_costI��;�)N+       ��K	�Ƈ��A�E*

logging/current_cost�;�^��+       ��K	���A�E*

logging/current_cost�}�;��X+       ��K	C#���A�E*

logging/current_cost�w�;�H�+       ��K	�S���A�E*

logging/current_cost�{�;�ݯ�+       ��K	M����A�E*

logging/current_cost.v�;3�}S+       ��K	����A�E*

logging/current_cost�s�;Y���+       ��K	ވ��A�E*

logging/current_cost u�;n�+       ��K	����A�E*

logging/current_costwu�;�%�d+       ��K	�?���A�E*

logging/current_cost,u�;�f+       ��K	yn���A�E*

logging/current_costcr�;Z��+       ��K	�����A�E*

logging/current_costxx�;8���+       ��K	�ω��A�E*

logging/current_costxs�;8Z�+       ��K	�����A�E*

logging/current_cost�o�;����+       ��K	�+���A�E*

logging/current_costIm�;� �X+       ��K	�X���A�E*

logging/current_cost�u�;���+       ��K	܈���A�E*

logging/current_cost�l�;�A��+       ��K	����A�E*

logging/current_cost�f�;���|+       ��K	�劔�A�E*

logging/current_cost?m�;���+       ��K	����A�E*

logging/current_costub�;;�+       ��K	$F���A�E*

logging/current_cost�f�;9U�+       ��K	ou���A�F*

logging/current_costYm�;9�+       ��K	�����A�F*

logging/current_costg�;p5�X+       ��K	�׋��A�F*

logging/current_cost�^�;�\o+       ��K	8���A�F*

logging/current_cost\�;ԕ�:+       ��K	�4���A�F*

logging/current_cost�\�;��CM+       ��K	<b���A�F*

logging/current_cost�[�;.+L�+       ��K	ِ���A�F*

logging/current_cost�V�;�N��+       ��K	�����A�F*

logging/current_cost�[�;�3�+       ��K	댔�A�F*

logging/current_cost�P�;�E�+       ��K	�>���A�F*

logging/current_costP�;悸�+       ��K	z���A�F*

logging/current_costP�;j�p{+       ��K	򫍔�A�F*

logging/current_costnP�;��)�+       ��K	�	���A�F*

logging/current_cost�P�;�2�+       ��K	�S���A�F*

logging/current_cost�K�;�8_�+       ��K	�����A�F*

logging/current_cost%T�;�+       ��K	Ύ��A�F*

logging/current_cost�O�;���+       ��K	P���A�F*

logging/current_cost�H�;I�H%+       ��K	�<���A�F*

logging/current_costdD�;!a!+       ��K	k����A�F*

logging/current_cost�H�;�yKV+       ��K	ޱ���A�F*

logging/current_cost(>�;o[+       ��K	Ꮤ�A�F*

logging/current_costD�;L'Ta+       ��K	����A�F*

logging/current_cost�F�;G���+       ��K	�U���A�F*

logging/current_costC@�;�uϴ+       ��K	~����A�F*

logging/current_cost�>�;Ίm+       ��K	�����A�F*

logging/current_costc9�;bw+       ��K	��A�F*

logging/current_cost*;�;Ty|]+       ��K	/���A�G*

logging/current_cost�6�;�w�o+       ��K	�f���A�G*

logging/current_cost�3�;��	+       ��K	8����A�G*

logging/current_costS1�;����+       ��K	֑��A�G*

logging/current_cost�8�;C��+       ��K	���A�G*

logging/current_costr1�;�9�m+       ��K	�A���A�G*

logging/current_cost:/�;Z_ۡ+       ��K	r���A�G*

logging/current_cost)�;P�0�+       ��K	5����A�G*

logging/current_cost0�;U�c+       ��K	�Ғ��A�G*

logging/current_costF'�;TO+       ��K	����A�G*

logging/current_cost�+�;��S0+       ��K	�;���A�G*

logging/current_cost��;�A��+       ��K	�n���A�G*

logging/current_cost�0�;��:+       ��K	�����A�G*

logging/current_cost$%�;��:�+       ��K	6㓔�A�G*

logging/current_cost�;��F�+       ��K	a���A�G*

logging/current_cost��;m���+       ��K	Q���A�G*

logging/current_cost��;wȬz+       ��K	o}���A�G*

logging/current_cost�;�]M�+       ��K	����A�G*

logging/current_cost��;z�+       ��K	ߔ��A�G*

logging/current_cost��;,�[+       ��K	-���A�G*

logging/current_cost� �;B��6+       ��K	�=���A�G*

logging/current_cost �;ʺ��+       ��K	Cn���A�G*

logging/current_cost�;��+       ��K	>����A�G*

logging/current_cost
�;ҟв+       ��K	\ŕ��A�G*

logging/current_cost��;�$�2+       ��K	����A�G*

logging/current_cost��;'[+       ��K	�&���A�G*

logging/current_costx �;�H��+       ��K	�U���A�H*

logging/current_cost��;��.J+       ��K	i����A�H*

logging/current_costB�;�@��+       ��K	�����A�H*

logging/current_costa�;m&+       ��K	�ߖ��A�H*

logging/current_cost"�;(�6+       ��K	����A�H*

logging/current_cost��;���+       ��K	C���A�H*

logging/current_cost!�;�Ug-+       ��K	�r���A�H*

logging/current_cost��;��[�+       ��K	M����A�H*

logging/current_cost��;��N�+       ��K	֗��A�H*

logging/current_cost��;Xb8+       ��K	����A�H*

logging/current_cost��; ��{+       ��K	�7���A�H*

logging/current_cost��;l�y+       ��K	�e���A�H*

logging/current_cost��;��ԟ+       ��K	�����A�H*

logging/current_cost��;��ZX+       ��K	�����A�H*

logging/current_cost�;�x�U+       ��K	V��A�H*

logging/current_cost2�;!�,+       ��K	{���A�H*

logging/current_costX�;j���+       ��K	�P���A�H*

logging/current_cost��;��%�+       ��K	q����A�H*

logging/current_cost��;s��+       ��K	�����A�H*

logging/current_costר;3���+       ��K	�ݙ��A�H*

logging/current_cost��;���"+       ��K	n���A�H*

logging/current_cost��;L��+       ��K	�I���A�H*

logging/current_cost��; ���+       ��K	�z���A�H*

logging/current_costv�;C�u�+       ��K	L����A�H*

logging/current_costb�;·��+       ��K	֚��A�H*

logging/current_cost��;��3�+       ��K	����A�I*

logging/current_costɈ;%� 9+       ��K	r2���A�I*

logging/current_cost�;9�6�+       ��K	1e���A�I*

logging/current_cost};]]y0+       ��K	Ò���A�I*

logging/current_cost��;��+       ��K	[Û��A�I*

logging/current_cost�z;�� {+       ��K	���A�I*

logging/current_cost��;���+       ��K	M���A�I*

logging/current_cost�;QB�p+       ��K	tP���A�I*

logging/current_cost\w;��+       ��K	$����A�I*

logging/current_cost�n;{"��+       ��K	�����A�I*

logging/current_cost}X;jI�+       ��K	Jۜ��A�I*

logging/current_costGl;D�Z+       ��K	!	���A�I*

logging/current_cost�P;?�+       ��K	�6���A�I*

logging/current_costxR;)ћ+       ��K	md���A�I*

logging/current_costzQ;�T+       ��K	2����A�I*

logging/current_costIw;nJ��+       ��K	W����A�I*

logging/current_costr\;��+       ��K	�Ꝕ�A�I*

logging/current_cost�[;�L=�+       ��K	����A�I*

logging/current_cost�<;]�+       ��K	
F���A�I*

logging/current_cost�G;��+       ��K	s���A�I*

logging/current_cost�=;��_+       ��K	1����A�I*

logging/current_cost�D;�"�+       ��K	�̞��A�I*

logging/current_cost�#;�yU+       ��K	z����A�I*

logging/current_cost�';�Ø+       ��K	/(���A�I*

logging/current_cost�;`tm�+       ��K	fU���A�I*

logging/current_costA;w+u6+       ��K	����A�I*

logging/current_cost�2;@39�+       ��K	m����A�J*

logging/current_cost�8;l���+       ��K	�۟��A�J*

logging/current_cost ;}[.�+       ��K	�	���A�J*

logging/current_cost!;���+       ��K	K?���A�J*

logging/current_costI;��۶+       ��K	�s���A�J*

logging/current_cost�;���+       ��K	����A�J*

logging/current_costq�~;��E+       ��K	�Р��A�J*

logging/current_cost;$S��+       ��K	����A�J*

logging/current_cost�;_kf+       ��K	$,���A�J*

logging/current_cost�;c�+       ��K	�Y���A�J*

logging/current_cost:�~;�s+       ��K	܅���A�J*

logging/current_costv�~;w���+       ��K	�����A�J*

logging/current_costk�~;�6��+       ��K	�ࡔ�A�J*

logging/current_cost�;q�JO+       ��K	G���A�J*

logging/current_cost��~;��ua+       ��K	<���A�J*

logging/current_cost�~;�bl�+       ��K	�i���A�J*

logging/current_cost��~;�0]2+       ��K	J����A�J*

logging/current_cost�~;]�+       ��K	�Ƣ��A�J*

logging/current_cost��~;iմ(+       ��K	����A�J*

logging/current_cost��~; �|&+       ��K	�!���A�J*

logging/current_cost�(;E�IW+       ��K	�Q���A�J*

logging/current_cost��~;���+       ��K	�����A�J*

logging/current_costi�~;H���+       ��K	�����A�J*

logging/current_cost2�~;J�vv+       ��K	@ޣ��A�J*

logging/current_cost%�~;a�Y+       ��K	���A�J*

logging/current_cost��~;��/�+       ��K	�<���A�K*

logging/current_costf�~;�F+       ��K	�j���A�K*

logging/current_cost��~;
n J+       ��K	K����A�K*

logging/current_cost!�~;���+       ��K	�̤��A�K*

logging/current_cost�~; u��+       ��K	����A�K*

logging/current_cost�~;X ��+       ��K	o7���A�K*

logging/current_costx�~;뾬\+       ��K	,l���A�K*

logging/current_cost��~;�e
,+       ��K	:����A�K*

logging/current_costz�~;��+       ��K	�ʥ��A�K*

logging/current_costۉ~;��<+       ��K	M����A�K*

logging/current_costx�~;����+       ��K	�4���A�K*

logging/current_costv�~;7�v+       ��K	�j���A�K*

logging/current_cost��~;l&*+       ��K	�����A�K*

logging/current_cost:�~;'�=(+       ��K	Z֦��A�K*

logging/current_cost4�~;��0�+       ��K	L���A�K*

logging/current_cost�p~;U��+       ��K	HE���A�K*

logging/current_cost�m~;|���+       ��K	y���A�K*

logging/current_cost�w~;�_s+       ��K	����A�K*

logging/current_costLN~;�G?F+       ��K	�ާ��A�K*

logging/current_cost�j~;�C�v+       ��K	����A�K*

logging/current_cost�P~;$�+       ��K	�B���A�K*

logging/current_cost5~;�{�*+       ��K	�r���A�K*

logging/current_cost�=~;���+       ��K	Ӧ���A�K*

logging/current_cost~?~;)�+       ��K	�֨��A�K*

logging/current_costc7~;i.�+       ��K	�
���A�K*

logging/current_cost)~;�}a�+       ��K	�@���A�K*

logging/current_costu~;#J��+       ��K	cr���A�L*

logging/current_cost�~;��(�+       ��K	?����A�L*

logging/current_cost�~;��+       ��K	Xԩ��A�L*

logging/current_cost�};���+       ��K	|���A�L*

logging/current_cost_~;���+       ��K	�5���A�L*

logging/current_cost��};.��+       ��K	k���A�L*

logging/current_costB�};q��j+       ��K	�����A�L*

logging/current_costx�};DS�+       ��K	�ͪ��A�L*

logging/current_cost�~;�x��+       ��K	�����A�L*

logging/current_cost�};�[)�+       ��K	�:���A�L*

logging/current_cost1~;�JM�+       ��K	�h���A�L*

logging/current_cost
~;;�P<+       ��K	�����A�L*

logging/current_cost��};���+       ��K	\Ϋ��A�L*

logging/current_cost��};����+       ��K	b ���A�L*

logging/current_cost��};O<y�+       ��K	F/���A�L*

logging/current_cost�~;R�`�+       ��K	9a���A�L*

logging/current_costӽ};���`+       ��K	ݒ���A�L*

logging/current_costɼ};	J +       ��K	�Ȭ��A�L*

logging/current_cost�};3�d+       ��K	k���A�L*

logging/current_costP�};7&'�+       ��K	=:���A�L*

logging/current_costk�};���+       ��K	Sy���A�L*

logging/current_cost[�};u�+�+       ��K	]����A�L*

logging/current_cost��};õ�1+       ��K	6䭔�A�L*

logging/current_cost��};�PO+       ��K	����A�L*

logging/current_costӦ};��,?+       ��K	�F���A�L*

logging/current_cost��};���&+       ��K	Gv���A�L*

logging/current_cost�};g�P�+       ��K	+����A�M*

logging/current_cost��};n�ڕ+       ��K	����A�M*

logging/current_costv};Z*I+       ��K	�,���A�M*

logging/current_cost`c};%7�U+       ��K	gr���A�M*

logging/current_cost�d};�ı$+       ��K	�����A�M*

logging/current_costb};v�O�+       ��K	Jݯ��A�M*

logging/current_cost	h};kZ�+       ��K	8���A�M*

logging/current_costkL};�S�+       ��K	�B���A�M*

logging/current_cost]N};)�mI+       ��K	0t���A�M*

logging/current_cost�y};��.'+       ��K	�����A�M*

logging/current_cost�i};��8+       ��K	�ذ��A�M*

logging/current_cost66};�$�+       ��K	+���A�M*

logging/current_cost�L};����+       ��K	*:���A�M*

logging/current_costJJ};	~ַ+       ��K	Hm���A�M*

logging/current_cost�};a?,�+       ��K	�����A�M*

logging/current_cost�};�P�g+       ��K	~ޱ��A�M*

logging/current_cost�#};M�x"+       ��K	���A�M*

logging/current_cost�};WF<}+       ��K	UF���A�M*

logging/current_cost�};�&Җ+       ��K	�w���A�M*

logging/current_costk	};��5n+       ��K	Y����A�M*

logging/current_cost�"};ޞ�+       ��K	�ܲ��A�M*

logging/current_cost20};�A�-+       ��K	����A�M*

logging/current_cost!�|;����+       ��K	�T���A�M*

logging/current_cost?�|;�P+       ��K	S����A�M*

logging/current_costS�|;�Kȋ+       ��K	�����A�M*

logging/current_cost1�|; )I+       ��K	�糔�A�N*

logging/current_cost��|;�A�+       ��K	����A�N*

logging/current_cost��|;m@c�+       ��K	oN���A�N*

logging/current_costY�|;M��