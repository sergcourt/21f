       �K"	  ���Abrain.Event:2���M�      ��	BJ���A"��
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
1layer_2/weights2/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_2/weights2*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_2/weights2/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_2/weights2*
valueB
 *׳]�
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
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
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
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
layer_3/weights3/readIdentitylayer_3/weights3*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
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
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
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

seed *
T0*"
_class
loc:@output/weights4*
seed2 *
dtype0*
_output_shapes

:
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
 output/biases4/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
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
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
_output_shapes
:*
T0*
out_type0
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
output/addcost/Placeholder'^train/gradients/cost/Mean_grad/truediv*'
_output_shapes
:���������*
T0
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
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
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
(train/gradients/layer_1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
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
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
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
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
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
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
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
-train/layer_3/weights3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_3/weights3*
valueB*    
�
train/layer_3/weights3/Adam
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
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
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
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
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
+train/output/biases4/Adam/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
train/output/biases4/Adam
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
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
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
train/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@layer_1/weights1
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
+train/Adam/update_layer_2/biases2/ApplyAdam	ApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@layer_2/biases2*
use_nesterov( *
_output_shapes
:*
use_locking( 
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
save/AssignAssignlayer_1/biases1save/RestoreV2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
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
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
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
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
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
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
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
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign"��#�     ��d]	�y���AJ܉
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

seed *
T0*#
_class
loc:@layer_1/weights1*
seed2 *
dtype0*
_output_shapes

:
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
/layer_2/weights2/Initializer/random_uniform/subSub/layer_2/weights2/Initializer/random_uniform/max/layer_2/weights2/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@layer_2/weights2
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
layer_2/ReluRelulayer_2/add*'
_output_shapes
:���������*
T0
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
!layer_3/biases3/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_3/biases3*
valueB*    
�
layer_3/biases3
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
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
z
layer_3/biases3/readIdentitylayer_3/biases3*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
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

seed *
T0*"
_class
loc:@output/weights4*
seed2 *
dtype0*
_output_shapes

:
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
 output/biases4/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
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
dtype0*
_output_shapes
:*
valueB"       
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
train/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
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
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
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
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
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
(train/gradients/layer_2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1
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
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
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
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
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
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
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
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
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
#train/layer_2/biases2/Adam_1/AssignAssigntrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
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
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_3/weights3*
valueB*    
�
train/layer_3/weights3/Adam_1
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
.train/output/weights4/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*"
_class
loc:@output/weights4*
valueB*    
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
train/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
U
train/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
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
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*"
_class
loc:@layer_1/biases1
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
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
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
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
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
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
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
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign""'
	summaries

logging/current_cost:0"�
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
output/biases4:0output/biases4/Assignoutput/biases4/read:02"output/biases4/Initializer/zeros:08"
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
train/output/biases4/Adam_1:0"train/output/biases4/Adam_1/Assign"train/output/biases4/Adam_1/read:02/train/output/biases4/Adam_1/Initializer/zeros:0�h��(       �pJ	��A*

logging/current_costP=o�X�*       ����	�<��A*

logging/current_costK�=<�U�*       ����	wn��A
*

logging/current_cost���<)�
�*       ����	����A*

logging/current_cost=��<y}�*       ����	:���A*

logging/current_costN�<2�3c*       ����	(���A*

logging/current_cost�J�<�b�4*       ����	P%��A*

logging/current_costa��<�0*       ����	�R��A#*

logging/current_cost�E�<���s*       ����	����A(*

logging/current_cost�8�<��&�*       ����	����A-*

logging/current_cost���<�� T*       ����	C���A2*

logging/current_cost���<�6l�*       ����	`	��A7*

logging/current_cost�M�<�
)F*       ����	�7��A<*

logging/current_cost�0�<dc*       ����	
e��AA*

logging/current_cost�\�<Ѿ*       ����	?���AF*

logging/current_cost���<���/*       ����	����AK*

logging/current_cost$>�<@��S*       ����	����AP*

logging/current_cost�ʮ<�=r�*       ����	@ ��AU*

logging/current_cost�S�<zO��*       ����	M��AZ*

logging/current_cost�ħ<��]�*       ����	�z��A_*

logging/current_cost��<[�8/*       ����	N���Ad*

logging/current_cost7�<��c
*       ����	����Ai*

logging/current_cost�.�<����*       ����	A��An*

logging/current_cost���<�CzT*       ����	�2��As*

logging/current_cost���<d�ń*       ����	3a��Ax*

logging/current_cost�<�y<�*       ����	����A}*

logging/current_cost�l�<����+       ��K	����A�*

logging/current_costf��<S�G�+       ��K	b���A�*

logging/current_cost���<���H+       ��K	���A�*

logging/current_cost@�v<�4�L+       ��K	�E��A�*

logging/current_cost�Kl<S �+       ��K	
r��A�*

logging/current_cost�a<�GF�+       ��K	*���A�*

logging/current_cost �V<rb��+       ��K	����A�*

logging/current_cost�L<�W�]+       ��K	� 	��A�*

logging/current_costAA<�$,�+       ��K	�,	��A�*

logging/current_costW�6<QQ63+       ��K	�[	��A�*

logging/current_cost�},<m�M�+       ��K	��	��A�*

logging/current_cost��"<l�]'+       ��K	V�	��A�*

logging/current_cost,3<�2!+       ��K	1�	��A�*

logging/current_cost7<)���+       ��K	�
��A�*

logging/current_cost�<��R�+       ��K	�@
��A�*

logging/current_cost�S�;��++       ��K	�p
��A�*

logging/current_cost]�;`�+       ��K	o�
��A�*

logging/current_cost���;*%+       ��K	��
��A�*

logging/current_cost+0�;��+       ��K	K�
��A�*

logging/current_cost���;R@+       ��K	�(��A�*

logging/current_costV��;T��+       ��K	FV��A�*

logging/current_costA"�;�!�+       ��K	���A�*

logging/current_cost�v�;����+       ��K	���A�*

logging/current_cost,�;�P@R+       ��K	����A�*

logging/current_cost�W�;��D+       ��K	{��A�*

logging/current_cost^��;��*+       ��K	4?��A�*

logging/current_cost>�;���j+       ��K	�k��A�*

logging/current_cost�ј;Y��D+       ��K	����A�*

logging/current_cost�Y�;5��+       ��K	����A�*

logging/current_cost=f�;��v+       ��K	����A�*

logging/current_cost~��;~��+       ��K	�&��A�*

logging/current_cost~��;R�Z+       ��K	�T��A�*

logging/current_cost�̐;�ވE+       ��K	����A�*

logging/current_cost��;EI�+       ��K	X���A�*

logging/current_cost7��;C�+       ��K	����A�*

logging/current_cost�5�;d5��+       ��K	���A�*

logging/current_cost)�;���+       ��K	v9��A�*

logging/current_cost\��;߂~�+       ��K	zg��A�*

logging/current_cost
��;�2r+       ��K	ǔ��A�*

logging/current_cost�^�;5��+       ��K	}���A�*

logging/current_cost=A�;{�o+       ��K	[���A�*

logging/current_cost(�;���+       ��K	��A�*

logging/current_costY�;�å&+       ��K	 L��A�*

logging/current_cost���;��+       ��K	my��A�*

logging/current_cost_�;`��P+       ��K	����A�*

logging/current_cost�؍;Md�+       ��K	x���A�*

logging/current_cost�č;���w+       ��K	�"��A�*

logging/current_cost���;#e��+       ��K	ri��A�*

logging/current_costN��;���+       ��K	ܬ��A�*

logging/current_costR��;%��+       ��K	���A�*

logging/current_costB��;���1+       ��K	�"��A�*

logging/current_cost�~�;l2Y�+       ��K	Y��A�*

logging/current_cost�r�;�ϲ+       ��K	ڠ��A�*

logging/current_costf�;ʃ��+       ��K	����A�*

logging/current_cost�Y�;=�+       ��K	��A�*

logging/current_cost�M�;��M�+       ��K	B^��A�*

logging/current_cost�@�;��=�+       ��K	I���A�*

logging/current_cost4�;��{r+       ��K	���A�*

logging/current_cost~&�;P�+       ��K	��A�*

logging/current_cost��;\&�e+       ��K	8��A�*

logging/current_costw
�;�=�L+       ��K	�r��A�*

logging/current_cost��;vBA+       ��K	����A�*

logging/current_costo�;E��+       ��K	J���A�*

logging/current_costy܌;D�r�+       ��K	[��A�*

logging/current_cost�̌;{��+       ��K	�G��A�*

logging/current_costŽ�;�Δ+       ��K	
|��A�*

logging/current_cost���;S#/+       ��K	_���A�*

logging/current_cost���;H��+       ��K	����A�*

logging/current_cost��;t<�+       ��K	,��A�*

logging/current_costՈ�;qV�+       ��K	�=��A�*

logging/current_cost�{�;��3^+       ��K	�l��A�*

logging/current_cost�n�;����+       ��K	���A�*

logging/current_cost�a�;�%}�+       ��K	����A�*

logging/current_cost�T�;N�>�+       ��K	���A�*

logging/current_cost�F�;��A�+       ��K	"2��A�*

logging/current_costw9�;'��+       ��K	�f��A�*

logging/current_cost9,�;㍀�+       ��K	����A�*

logging/current_cost1�;��m+       ��K	����A�*

logging/current_cost��;g@iD+       ��K	� ��A�*

logging/current_cost-�;Iɀ�+       ��K	Tk��A�*

logging/current_costx��;�l�A+       ��K	����A�*

logging/current_costi��;����+       ��K	b���A�*

logging/current_cost�;.p+       ��K	���A�*

logging/current_cost
܋;�c"+       ��K	�J��A�*

logging/current_cost�ҋ;�e]+       ��K	�|��A�*

logging/current_cost�ɋ;Aڐ�+       ��K	����A�*

logging/current_cost��;��*c+       ��K	����A�*

logging/current_costQ��;��W+       ��K	�$��A�*

logging/current_cost:��;����+       ��K	�f��A�*

logging/current_cost쩋;1�I�+       ��K	����A�*

logging/current_costF��;�d��+       ��K	%���A�*

logging/current_cost5��;�K�+       ��K	\��A�*

logging/current_costГ�;�N�+       ��K	�J��A�*

logging/current_cost�;��+       ��K	*���A�*

logging/current_cost���;�+       ��K	���A�*

logging/current_costF�;pt(�+       ��K	x���A�*

logging/current_costx�;@(�.+       ��K	U��A�*

logging/current_costKq�;i&�g+       ��K	�B��A�*

logging/current_costWj�;���+       ��K	x��A�*

logging/current_cost�c�;H�dc+       ��K	���A�*

logging/current_cost�\�;ҟ�+       ��K	����A�*

logging/current_cost�V�;���+       ��K	W��A�*

logging/current_cost�P�;�(�+       ��K	$5��A�*

logging/current_costWJ�;�\0+       ��K	Yb��A�*

logging/current_cost^D�;�0R++       ��K	ӓ��A�*

logging/current_cost�>�;�{�+       ��K	+���A�*

logging/current_cost�8�;���+       ��K	����A�*

logging/current_cost2�;S���+       ��K	�.��A�*

logging/current_cost,�;3p&�+       ��K	�_��A�*

logging/current_cost&�;� I�+       ��K	h���A�*

logging/current_cost��;�Xվ+       ��K	����A�*

logging/current_cost��;K�+       ��K	~	��A�*

logging/current_cost��;F"-�+       ��K	S=��A�*

logging/current_cost��;��w{+       ��K	�w��A�*

logging/current_cost��;	�~3+       ��K	����A�*

logging/current_cost��;��M+       ��K	����A�*

logging/current_cost���;J�@+       ��K	�>��A�*

logging/current_cost���;��fE+       ��K	)���A�*

logging/current_cost�;9`l�+       ��K	y  ��A�*

logging/current_cost1�;lq�+       ��K	�Z ��A�*

logging/current_cost�;-<E+       ��K	7� ��A�*

logging/current_costO܊;�{�}+       ��K	(� ��A�*

logging/current_costV֊;#��+       ��K	0	!��A�*

logging/current_costLЊ;���+       ��K	�=!��A�*

logging/current_cost_ʊ;��k�+       ��K	�{!��A�*

logging/current_cost]Ċ;Fؘ�+       ��K	 �!��A�*

logging/current_costh��;��ͤ+       ��K	z�!��A�*

logging/current_cost���;��;+       ��K	 ("��A�*

logging/current_costײ�;��+       ��K	�b"��A�*

logging/current_cost���;W��}+       ��K	V�"��A�*

logging/current_cost֦�;mm�P+       ��K	�"��A�*

logging/current_costP��;��R9+       ��K	�#��A�*

logging/current_costr��;����+       ��K	.=#��A�*

logging/current_cost}��;�%G�+       ��K	�q#��A�*

logging/current_costV��;k%
k+       ��K	"�#��A�*

logging/current_cost��;�u_�+       ��K	�$��A�*

logging/current_cost���;�!��+       ��K	�V$��A�*

logging/current_cost/�;��e+       ��K	̕$��A�*

logging/current_costny�;U#y�+       ��K	��$��A�*

logging/current_costs�;4�+       ��K	�%��A�*

logging/current_costVm�;��E+       ��K	(T%��A�*

logging/current_cost@h�;)�A+       ��K	Q�%��A�*

logging/current_cost�b�;#�C�+       ��K	i�%��A�*

logging/current_cost�\�;�0)�+       ��K	��%��A�*

logging/current_cost�V�;6)��+       ��K	�7&��A�*

logging/current_costQ�;a��+       ��K	2g&��A�*

logging/current_cost�K�;��J�+       ��K	5�&��A�*

logging/current_costAF�;�_�G+       ��K	��&��A�*

logging/current_costQ@�;q��+       ��K	�'��A�*

logging/current_cost�:�;p��+       ��K	B'��A�*

logging/current_cost@5�;���d+       ��K	�q'��A�*

logging/current_cost�/�;1��+       ��K	Υ'��A�*

logging/current_cost+*�;� �h+       ��K	$�'��A�*

logging/current_cost�$�;���Z+       ��K	�(��A�*

logging/current_cost��;� 6�+       ��K	4G(��A�*

logging/current_cost�;����+       ��K	{z(��A�*

logging/current_costQ�;��f+       ��K	�(��A�*

logging/current_cost��;�#�N+       ��K	�(��A�*

logging/current_cost�	�;�XM+       ��K	�)��A�*

logging/current_costk�;����+       ��K	�J)��A�*

logging/current_costM��;�	+       ��K	#�)��A�*

logging/current_costf��;;Y+       ��K	��)��A�*

logging/current_cost�;��+       ��K	��)��A�*

logging/current_cost��;���+       ��K	�!*��A�*

logging/current_cost`�;���+       ��K	(S*��A�*

logging/current_cost��;���+       ��K	��*��A�*

logging/current_costH߉;���}+       ��K	q�*��A�*

logging/current_cost�ى;���+       ��K	��*��A�*

logging/current_costԉ;'	6+       ��K	$.+��A�*

logging/current_cost�Ή;��++       ��K	�b+��A�*

logging/current_cost�ɉ;h,�-+       ��K	N�+��A�*

logging/current_cost�É;�V�+       ��K	c�+��A�*

logging/current_cost���;)���+       ��K	x,��A�*

logging/current_cost���;9��+       ��K		D,��A�*

logging/current_cost��;�7S�+       ��K	�u,��A�*

logging/current_cost���;�"�f+       ��K	�,��A�*

logging/current_cost���;Ȑ((+       ��K	�-��A�*

logging/current_cost���;��4+       ��K	W2-��A�*

logging/current_cost5��;h��+       ��K	�i-��A�*

logging/current_cost���;ӊ��+       ��K	��-��A�*

logging/current_cost���;�wB�+       ��K	��-��A�*

logging/current_costφ�;줟�+       ��K	�$.��A�*

logging/current_cost.��;sF	+       ��K	�X.��A�*

logging/current_costOz�;�\3�+       ��K	��.��A�*

logging/current_cost$t�;B�c+       ��K	:�.��A�*

logging/current_cost�m�;+`�#+       ��K	�/��A�*

logging/current_cost�g�;�ת*+       ��K	�</��A�*

logging/current_cost\b�;p�+       ��K	ru/��A�*

logging/current_cost�]�;e���+       ��K	��/��A�*

logging/current_costsW�;]ELf+       ��K	J�/��A�*

logging/current_costR�;�dٹ+       ��K	�&0��A�*

logging/current_cost!M�;�J}_+       ��K	�e0��A�*

logging/current_cost�G�;��nx+       ��K	Y�0��A�*

logging/current_cost9B�;��h+       ��K	��0��A�*

logging/current_cost{=�;98h�+       ��K	1��A�*

logging/current_cost�8�;`V�+       ��K	_X1��A�*

logging/current_cost+3�;'�o2+       ��K	(�1��A�*

logging/current_cost%.�;p��l+       ��K	��1��A�*

logging/current_costY)�;!
�+       ��K	|2��A�*

logging/current_cost3$�;k�x�+       ��K	�W2��A�*

logging/current_costS�;���+       ��K	��2��A�*

logging/current_cost��;R�-8+       ��K	��2��A�*

logging/current_costs�;h��\+       ��K	�"3��A�*

logging/current_coste�;�4�[+       ��K	�i3��A�*

logging/current_cost��;���+       ��K	�3��A�*

logging/current_cost��;�$7i+       ��K	��3��A�	*

logging/current_cost�;L�v�+       ��K	�4��A�	*

logging/current_cost��;6le+       ��K	wJ4��A�	*

logging/current_cost���;���+       ��K	l�4��A�	*

logging/current_costf�;�PA�+       ��K	��4��A�	*

logging/current_costz�;+��+       ��K	��4��A�	*

logging/current_cost��;����+       ��K	3;5��A�	*

logging/current_cost�;{[Ď+       ��K	��5��A�	*

logging/current_cost��;]�ԯ+       ��K	��5��A�	*

logging/current_cost�݈;i�P�+       ��K	X�5��A�	*

logging/current_costو;�F�+       ��K	O;6��A�	*

logging/current_cost'Ո;WQL+       ��K	�o6��A�	*

logging/current_cost�Ј;�,)�+       ��K	Ϧ6��A�	*

logging/current_costM̈;��,&+       ��K	�6��A�	*

logging/current_costoǈ;>�>^+       ��K	7��A�	*

logging/current_costCÈ;��](+       ��K	FM7��A�	*

logging/current_cost��;b9I�+       ��K	�7��A�	*

logging/current_cost$��;q�W+       ��K	��7��A�	*

logging/current_cost!��;���+       ��K	u8��A�	*

logging/current_cost���;=Jo+       ��K	MX8��A�	*

logging/current_cost���;sU��+       ��K	R�8��A�	*

logging/current_costJ��;�M+       ��K	��8��A�	*

logging/current_costh��;"���+       ��K	�%9��A�	*

logging/current_cost��;�Ͼ+       ��K	+n9��A�	*

logging/current_costM��;�i�Z+       ��K	�9��A�	*

logging/current_costS��;^�+       ��K	��9��A�
*

logging/current_cost���;D_�+       ��K	::��A�
*

logging/current_costX��;�2?+       ��K	�N:��A�
*

logging/current_cost덈;���+       ��K	��:��A�
*

logging/current_cost`��;�F�$+       ��K	��:��A�
*

logging/current_cost���;��4�+       ��K	n;��A�
*

logging/current_cost���;ʯ+       ��K	\G;��A�
*

logging/current_cost�;p�P+       ��K	t�;��A�
*

logging/current_cost {�;��f+       ��K	=<��A�
*

logging/current_cost*x�;�"G�+       ��K	K<��A�
*

logging/current_costt�;ʆ�+       ��K	ȓ<��A�
*

logging/current_cost�o�;J�P+       ��K	��<��A�
*

logging/current_cost�l�;u�C�+       ��K	�6=��A�
*

logging/current_cost�h�;�'g+       ��K	,{=��A�
*

logging/current_cost�e�;�gä+       ��K	��=��A�
*

logging/current_costb�;<�I|+       ��K	��=��A�
*

logging/current_costS^�;$p2+       ��K	�B>��A�
*

logging/current_cost[�;!1�+       ��K	5�>��A�
*

logging/current_costoW�;5��"+       ��K	ɶ>��A�
*

logging/current_cost7T�;A;��+       ��K	��>��A�
*

logging/current_cost�P�;��h+       ��K	 ?��A�
*

logging/current_cost�M�;);n+       ��K	|^?��A�
*

logging/current_cost0J�;3�q+       ��K	��?��A�
*

logging/current_cost�F�;u�ɛ+       ��K	��?��A�
*

logging/current_cost�C�;ِ��+       ��K	�@��A�
*

logging/current_costP@�;BV�I+       ��K	(F@��A�
*

logging/current_costW=�;��-�+       ��K	�y@��A�*

logging/current_cost�9�;�ˍT+       ��K	�@��A�*

logging/current_cost 7�;	�x*+       ��K	��@��A�*

logging/current_cost�3�;���+       ��K	bA��A�*

logging/current_cost�0�;2�n+       ��K	WA��A�*

logging/current_cost�-�;ھ�@+       ��K	~�A��A�*

logging/current_cost�*�;���+       ��K	J�A��A�*

logging/current_cost�'�;���+       ��K	��A��A�*

logging/current_cost�$�;��۫+       ��K	,B��A�*

logging/current_cost�!�;@�";+       ��K	�qB��A�*

logging/current_cost��;T���+       ��K	��B��A�*

logging/current_cost��;a2b�+       ��K	��B��A�*

logging/current_cost��;
�[l+       ��K	C��A�*

logging/current_costf�;p#�+       ��K	�gC��A�*

logging/current_costz�;���/+       ��K	��C��A�*

logging/current_cost��;�]#�+       ��K	$�C��A�*

logging/current_cost��;暽�+       ��K	�@D��A�*

logging/current_cost�;�ʚ�+       ��K	�}D��A�*

logging/current_costE�;���I+       ��K	M�D��A�*

logging/current_cost��;ϵO%+       ��K	��D��A�*

logging/current_cost�;pbV^+       ��K	,7E��A�*

logging/current_costW �;�*Xa+       ��K	mE��A�*

logging/current_cost���;0j��+       ��K	��E��A�*

logging/current_cost��;����+       ��K	��E��A�*

logging/current_cost��;�"N+       ��K	�F��A�*

logging/current_cost��;��H�+       ��K	6@F��A�*

logging/current_cost�;\�4#+       ��K	W{F��A�*

logging/current_cost �;]�+       ��K	�F��A�*

logging/current_cost+�;_ Hs+       ��K	A�F��A�*

logging/current_cost�;��̹+       ��K	G��A�*

logging/current_cost�;\�+       ��K	0HG��A�*

logging/current_cost��;QO3�+       ��K	{G��A�*

logging/current_cost�߇;�A"+       ��K	��G��A�*

logging/current_cost�܇;uܶ�+       ��K	|�G��A�*

logging/current_cost�ه;�&:Z+       ��K	;H��A�*

logging/current_cost�Շ;�DC+       ��K	�gH��A�*

logging/current_cost�ч;X;��+       ��K	ٝH��A�*

logging/current_cost ·;��!�+       ��K	��H��A�*

logging/current_costrɇ;,�+       ��K	eI��A�*

logging/current_cost�ć;,-2+       ��K	8BI��A�*

logging/current_cost��;���M+       ��K	p�I��A�*

logging/current_costϸ�;�4�a+       ��K	/�I��A�*

logging/current_cost!��;v��J+       ��K	TJ��A�*

logging/current_cost6��;밟k+       ��K	�IJ��A�*

logging/current_cost�;�P+       ��K	��J��A�*

logging/current_costN��;�e|+       ��K	K��A�*

logging/current_cost���;FW~+       ��K	FK��A�*

logging/current_cost��;V���+       ��K	:�K��A�*

logging/current_cost�;N�Q$+       ��K	ýK��A�*

logging/current_cost��;p��D+       ��K	_�K��A�*

logging/current_cost�y�;��+�+       ��K	�9L��A�*

logging/current_cost�u�;��u2+       ��K	�|L��A�*

logging/current_costUo�;E��+       ��K	^�L��A�*

logging/current_costji�;�<�+       ��K	r�L��A�*

logging/current_cost�a�;G�m+       ��K	�+M��A�*

logging/current_costOY�;���*+       ��K	U`M��A�*

logging/current_cost�P�;�}~�+       ��K	
�M��A�*

logging/current_cost2I�;(��+       ��K	�M��A�*

logging/current_cost�A�;o�p�+       ��K	�M��A�*

logging/current_cost�:�;�2��+       ��K	,N��A�*

logging/current_cost�4�;��+       ��K	�fN��A�*

logging/current_costY/�;��T�+       ��K	��N��A�*

logging/current_cost�)�;�6;+       ��K	��N��A�*

logging/current_costi$�;(��4+       ��K	O��A�*

logging/current_cost�;�Ku]+       ��K	sFO��A�*

logging/current_cost��;��	+       ��K	{xO��A�*

logging/current_cost��;��5+       ��K	4�O��A�*

logging/current_cost{�;>��+       ��K	vCP��A�*

logging/current_costk	�;̚y[+       ��K	�P��A�*

logging/current_cost��;���+       ��K	?�P��A�*

logging/current_cost���;��y+       ��K	��P��A�*

logging/current_cost`��;(�*+       ��K	;+Q��A�*

logging/current_cost��;��U+       ��K	McQ��A�*

logging/current_cost��;�-�'+       ��K	2�Q��A�*

logging/current_cost�;ب�+       ��K	��Q��A�*

logging/current_cost&��;}Z�+       ��K	R��A�*

logging/current_cost�چ;�Am+       ��K	P>R��A�*

logging/current_cost�׆;14+       ��K	/wR��A�*

logging/current_cost҆;�QZ+       ��K	m�R��A�*

logging/current_costC͆;_���+       ��K	!�R��A�*

logging/current_cost�Ȇ;P�wE+       ��K	�S��A�*

logging/current_cost�;�T��+       ��K	�\S��A�*

logging/current_costU��;Lݫ+       ��K	ЗS��A�*

logging/current_costص�;����+       ��K	,�S��A�*

logging/current_cost��;���+       ��K	�	T��A�*

logging/current_cost���;d�ʗ+       ��K	�BT��A�*

logging/current_costD��;�Pl+       ��K	��T��A�*

logging/current_cost���; �.+       ��K	"�T��A�*

logging/current_cost���;�{�+       ��K	*�T��A�*

logging/current_cost��;�)@�+       ��K	�5U��A�*

logging/current_costs��;mI�Z+       ��K	�tU��A�*

logging/current_cost���;���O+       ��K	��U��A�*

logging/current_costB��;�U�+       ��K	|�U��A�*

logging/current_cost逆;k�x+       ��K	N*V��A�*

logging/current_cost{�;�K+       ��K	�lV��A�*

logging/current_cost�v�;�*�e+       ��K	��V��A�*

logging/current_cost�q�;@�W+       ��K	�V��A�*

logging/current_costRn�;���+       ��K	�W��A�*

logging/current_cost�h�;���&+       ��K	YW��A�*

logging/current_cost�d�;� A+       ��K	��W��A�*

logging/current_costa�;��q+       ��K	j�W��A�*

logging/current_cost�\�;�^�+       ��K	n!X��A�*

logging/current_costSV�;>jb�+       ��K	�dX��A�*

logging/current_cost�R�;ˎ�V+       ��K	�X��A�*

logging/current_cost�M�;���+       ��K	��X��A�*

logging/current_costqI�;A���+       ��K	�Y��A�*

logging/current_costnF�;s`iX+       ��K	�CY��A�*

logging/current_cost�A�;<��+       ��K	؀Y��A�*

logging/current_costq>�;���B+       ��K	7�Y��A�*

logging/current_cost�9�;���+       ��K	��Y��A�*

logging/current_cost6�;vy��+       ��K	.(Z��A�*

logging/current_cost�2�;ණN+       ��K	�`Z��A�*

logging/current_cost7/�;Ԗ��+       ��K	��Z��A�*

logging/current_cost�+�;�B�+       ��K	I�Z��A�*

logging/current_cost'�;�2� +       ��K	�[��A�*

logging/current_cost�"�;��z+       ��K	~U[��A�*

logging/current_costS�;��E
+       ��K	��[��A�*

logging/current_cost��;m6�+       ��K	��[��A�*

logging/current_cost��;�;�+       ��K	�
\��A�*

logging/current_costB�;}�De+       ��K	�;\��A�*

logging/current_costK�;���+       ��K	�p\��A�*

logging/current_costc�;�L#Y+       ��K	Χ\��A�*

logging/current_cost�;_^+       ��K	<�\��A�*

logging/current_cost���;/�=|+       ��K	]��A�*

logging/current_cost���;�Ta�+       ��K	�D]��A�*

logging/current_costj��;#N+       ��K	�y]��A�*

logging/current_cost��;�Q�+       ��K	��]��A�*

logging/current_costj�;�$x�+       ��K	��]��A�*

logging/current_cost�;�T�x+       ��K	(^��A�*

logging/current_cost�;X�m�+       ��K	!g^��A�*

logging/current_cost��;ꮗ�+       ��K	��^��A�*

logging/current_costYޅ;S���+       ��K	��^��A�*

logging/current_cost�م;S�d�+       ��K	�_��A�*

logging/current_cost�Ѕ;���+       ��K	I_��A�*

logging/current_costƅ;^��#+       ��K	Y|_��A�*

logging/current_costn��;��N+       ��K	K�_��A�*

logging/current_cost���;����+       ��K	~`��A�*

logging/current_costp��;���+       ��K	dY`��A�*

logging/current_costY��;ȋ�+       ��K	�`��A�*

logging/current_cost���;*8�4+       ��K	��`��A�*

logging/current_cost]v�;/X��+       ��K	ba��A�*

logging/current_cost�m�;Bm�o+       ��K	ZPa��A�*

logging/current_cost8e�;��¾+       ��K	@�a��A�*

logging/current_cost9[�;���+       ��K	o�a��A�*

logging/current_cost�T�;�R��+       ��K	��a��A�*

logging/current_cost�O�;����+       ��K	@b��A�*

logging/current_costLJ�;5rIT+       ��K	Ib��A�*

logging/current_cost�G�;^L�+       ��K	Hzb��A�*

logging/current_cost|B�;�T�f+       ��K	��b��A�*

logging/current_cost>�;�"6+       ��K	��b��A�*

logging/current_cost-:�;��c+       ��K	�c��A�*

logging/current_costL8�;[��+       ��K	�Ec��A�*

logging/current_cost�4�;���t+       ��K	~c��A�*

logging/current_cost�0�;k���+       ��K	n�c��A�*

logging/current_cost�-�;�z�+       ��K	8�c��A�*

logging/current_costF+�;�.�*+       ��K	�d��A�*

logging/current_cost])�;�c�+       ��K	o?d��A�*

logging/current_cost$�;L���+       ��K	erd��A�*

logging/current_cost$"�;"��+       ��K	M�d��A�*

logging/current_cost5�;:��+       ��K	��d��A�*

logging/current_cost<�;AG�+       ��K	{!e��A�*

logging/current_cost��;�1�\+       ��K	�Ue��A�*

logging/current_cost��;�NCR+       ��K	F�e��A�*

logging/current_cost��;��{q+       ��K	��e��A�*

logging/current_cost��;`��m+       ��K	5&f��A�*

logging/current_cost�;qC5�+       ��K	�cf��A�*

logging/current_cost��;Dx+       ��K	F�f��A�*

logging/current_cost�
�;�7u+       ��K	��f��A�*

logging/current_cost
�;���+       ��K	�;g��A�*

logging/current_cost��;�N�+       ��K	tvg��A�*

logging/current_cost;�;G{��+       ��K	��g��A�*

logging/current_cost�;��+       ��K	s�g��A�*

logging/current_cost��;�)
#+       ��K	J"h��A�*

logging/current_cost~��;�I�=+       ��K	�Th��A�*

logging/current_cost���;�̺+       ��K	��h��A�*

logging/current_costE��;@f��+       ��K	$�h��A�*

logging/current_costJ��;{+       ��K	k�h��A�*

logging/current_cost���;��++       ��K	�i��A�*

logging/current_cost���;g\+       ��K	�^i��A�*

logging/current_cost��;3vn�+       ��K	ˑi��A�*

logging/current_cost��;Z�j�+       ��K	u�i��A�*

logging/current_cost��;ą�+       ��K	��i��A�*

logging/current_costu��;��F +       ��K	2j��A�*

logging/current_costi�;{M�+       ��K	eij��A�*

logging/current_costZ�;7iw�+       ��K	B�j��A�*

logging/current_cost��;�Y�o+       ��K	��j��A�*

logging/current_cost��;���\+       ��K	�k��A�*

logging/current_cost��;z*�c+       ��K	�<k��A�*

logging/current_costq�;�Bć+       ��K	Aik��A�*

logging/current_cost!�;��b"+       ��K	��k��A�*

logging/current_cost��;�"�Z+       ��K	�k��A�*

logging/current_cost��;��{(+       ��K	0l��A�*

logging/current_cost��;n�B+       ��K	>6l��A�*

logging/current_costj�;�1�#+       ��K	�fl��A�*

logging/current_costW�;Da+       ��K	#�l��A�*

logging/current_cost��;`��+       ��K	
�l��A�*

logging/current_cost��;S��M+       ��K	�m��A�*

logging/current_cost��;�I+       ��K	�5m��A�*

logging/current_costQ�;c�D+       ��K	^am��A�*

logging/current_cost@߄;�*SE+       ��K	\�m��A�*

logging/current_cost��;ք��+       ��K	��m��A�*

logging/current_cost ބ;��iI+       ��K	�n��A�*

logging/current_costEބ;����+       ��K	W@n��A�*

logging/current_cost�݄;��X�+       ��K	�pn��A�*

logging/current_cost�܄;��+       ��K	�n��A�*

logging/current_costN܄;���+       ��K	��n��A�*

logging/current_cost�ۄ;\��+       ��K	�o��A�*

logging/current_cost�ڄ;^MK�+       ��K	�So��A�*

logging/current_cost2ڄ;��J+       ��K	8�o��A�*

logging/current_cost0ل;̈b+       ��K	*�o��A�*

logging/current_costr؄;����+       ��K	up��A�*

logging/current_cost�؄;b@h+       ��K	aHp��A�*

logging/current_cost�ׄ;"�+       ��K	��p��A�*

logging/current_cost؄;�HJ+       ��K	:�p��A�*

logging/current_costQք;�<X+       ��K	��p��A�*

logging/current_cost�Մ; Y~+       ��K	�,q��A�*

logging/current_costք;��[�+       ��K	Jjq��A�*

logging/current_cost�ք;#:	+       ��K	��q��A�*

logging/current_costTՄ;���+       ��K	��q��A�*

logging/current_cost\Մ;��+       ��K	#r��A�*

logging/current_costYՄ;#h��+       ��K	Gr��A�*

logging/current_cost�ӄ;'�R�+       ��K	�zr��A�*

logging/current_costՄ;ן+       ��K	�r��A�*

logging/current_cost�ӄ;��1�+       ��K	��r��A�*

logging/current_cost�҄;��۬+       ��K	>s��A�*

logging/current_cost�ӄ;z��+       ��K	\�s��A�*

logging/current_cost@ӄ;�1T+       ��K	Ʒs��A�*

logging/current_costEӄ;���+       ��K	 �s��A�*

logging/current_costB҄;��I�+       ��K	�6t��A�*

logging/current_cost4ф;��x+       ��K	ctt��A�*

logging/current_costuф;�`+       ��K	>�t��A�*

logging/current_cost�Є;@���+       ��K	~�t��A�*

logging/current_cost�Є;5�_+       ��K	=u��A�*

logging/current_cost{ф;��[�+       ��K	Ξu��A�*

logging/current_cost4҄;w�RI+       ��K	R�u��A�*

logging/current_cost@ф;�W@�+       ��K	��u��A�*

logging/current_cost�Є;껭o+       ��K	�0v��A�*

logging/current_cost�τ;��1+       ��K	�gv��A�*

logging/current_costjЄ;��-+       ��K	�v��A�*

logging/current_cost�Є;a��z+       ��K	o�v��A�*

logging/current_cost�τ;v�"?+       ��K	w�v��A�*

logging/current_cost�τ;�2y+       ��K	|<w��A�*

logging/current_cost�΄;g�+       ��K	�{w��A�*

logging/current_costIτ;����+       ��K	P�w��A�*

logging/current_cost�΄;J�r+       ��K	��w��A�*

logging/current_costτ;2�go+       ��K	�*x��A�*

logging/current_costτ;<^��+       ��K	Ebx��A�*

logging/current_cost�τ;��B�+       ��K	s�x��A�*

logging/current_cost�τ;�a)+       ��K	��x��A�*

logging/current_cost�Є;��+       ��K	T�x��A�*

logging/current_cost�̈́;��̀+       ��K	u#y��A�*

logging/current_cost�΄;m ^�+       ��K	�Py��A�*

logging/current_costx΄;A�͐+       ��K	��y��A�*

logging/current_cost.΄;t�oa+       ��K	y�y��A�*

logging/current_cost%΄;gp�+       ��K	7�y��A�*

logging/current_cost�̄;#ƈ�+       ��K	�z��A�*

logging/current_cost�̈́;KX��+       ��K	�Qz��A�*

logging/current_cost�̄;w��+       ��K	�z��A�*

logging/current_cost+̈́;�kV+       ��K	��z��A�*

logging/current_cost̄;����+       ��K	]�z��A�*

logging/current_cost�̈́;ӗy+       ��K	�{��A�*

logging/current_costJ̄;�E�+       ��K	�K{��A�*

logging/current_cost̄;��.+       ��K	\�{��A�*

logging/current_cost�̈́;w��+       ��K	��{��A�*

logging/current_cost�̄;}��+       ��K	�|��A�*

logging/current_cost�̄;��;�+       ��K	�J|��A�*

logging/current_cost�̄;��)+       ��K	��|��A�*

logging/current_cost�̄;�}��+       ��K	��|��A�*

logging/current_cost̄;��m+       ��K	E|}��A�*

logging/current_costB̄;�I�+       ��K	�}��A�*

logging/current_cost�̈́;�F+       ��K	��}��A�*

logging/current_cost˄;g�}+       ��K	d6~��A�*

logging/current_cost�ʄ;%�N�+       ��K	�o~��A�*

logging/current_costd˄;���+       ��K	��~��A�*

logging/current_costUʄ;�R�+       ��K	4�~��A�*

logging/current_cost�ʄ;+j�s+       ��K	���A�*

logging/current_cost�̄;<o�+       ��K	H��A�*

logging/current_cost2ʄ;��ڹ+       ��K	����A�*

logging/current_cost;˄;�</[+       ��K	����A�*

logging/current_cost%ʄ;A�U+       ��K	,���A�*

logging/current_cost�ʄ;1I��+       ��K	�!���A�*

logging/current_cost+ʄ;>(�+       ��K	V���A�*

logging/current_costz˄;  ��+       ��K	�����A�*

logging/current_cost˄;cph+       ��K	n����A�*

logging/current_cost�ʄ; W�+       ��K		���A�*

logging/current_cost�ʄ;��+       ��K	l>���A�*

logging/current_costlɄ;]v�'+       ��K	�q���A�*

logging/current_costL̄;�^�y+       ��K	�����A�*

logging/current_cost�Ʉ;��+       ��K	*؁��A�*

logging/current_cost�Ʉ;"�w�+       ��K	l+���A�*

logging/current_costW˄;q�+       ��K	/e���A�*

logging/current_costWɄ;I���+       ��K	�����A�*

logging/current_cost�Ȅ;a�cp+       ��K	�Ƃ��A�*

logging/current_cost�Ȅ;�&+       ��K	���A�*

logging/current_cost�ʄ;���+       ��K	�J���A�*

logging/current_cost�Ȅ;** �+       ��K	��A�*

logging/current_costɄ;A�+       ��K	�����A�*

logging/current_cost�Ʉ;�D��+       ��K	����A�*

logging/current_cost"Ʉ;���+       ��K	�<���A�*

logging/current_costɄ;��&+       ��K	!m���A�*

logging/current_cost�Ʉ;��н+       ��K	Ϣ���A�*

logging/current_cost&ʄ;E̤�+       ��K	�Մ��A�*

logging/current_cost�Ʉ;��G+       ��K	�
���A�*

logging/current_costsȄ;֜}U+       ��K	9@���A�*

logging/current_cost�Ȅ;�Ǌ+       ��K	�u���A�*

logging/current_cost[Ȅ; �ȕ+       ��K	�����A�*

logging/current_cost)Ʉ;�	�+       ��K	F慆�A�*

logging/current_costyȄ;�Do�+       ��K	"#���A�*

logging/current_costwȄ;#��+       ��K	TV���A�*

logging/current_cost�Ȅ;��?�+       ��K	����A�*

logging/current_cost�Ǆ;�Q�+       ��K	����A�*

logging/current_cost�Ǆ;,BIk+       ��K	醆�A�*

logging/current_costwǄ;�o�+       ��K	����A�*

logging/current_costɄ;b[+       ��K	�H���A�*

logging/current_cost�Ǆ;&8<E+       ��K	�w���A�*

logging/current_costLǄ;�|+       ��K	-����A�*

logging/current_cost0Ȅ;�[+       ��K	�܇��A�*

logging/current_costȄ;偾�+       ��K	����A�*

logging/current_cost�Ƅ;�B�
+       ��K	A8���A�*

logging/current_cost�Ƅ;I�8+       ��K	3k���A�*

logging/current_cost�Ǆ;�s("+       ��K	�����A�*

logging/current_costǄ;l>\-+       ��K	`∆�A�*

logging/current_cost�Ȅ;�
+       ��K	E!���A�*

logging/current_cost,ʄ;���+       ��K	+a���A�*

logging/current_cost�Ǆ;F�0+       ��K	A����A�*

logging/current_cost�Ǆ;�~�+       ��K	�ቆ�A�*

logging/current_cost:Ȅ;�t��+       ��K	�&���A�*

logging/current_cost�Ƅ;�/`�+       ��K	ya���A�*

logging/current_costuȄ;�j��+       ��K	�����A�*

logging/current_cost�Ǆ;���+       ��K	�ϊ��A�*

logging/current_cost�Ƅ;ƞ�r+       ��K	p���A�*

logging/current_costƄ;g*S�+       ��K	�<���A�*

logging/current_costsƄ;@iY+       ��K	As���A�*

logging/current_cost�ń;�t�+       ��K	�����A�*

logging/current_costƄ;F��+       ��K	�싆�A�*

logging/current_cost�Ǆ;���+       ��K	�&���A�*

logging/current_costNƄ;��R�+       ��K	�`���A�*

logging/current_cost=ń;�@�+       ��K	E����A�*

logging/current_cost�ń;�:9+       ��K	�ό��A�*

logging/current_costVń;s[2+       ��K	M���A�*

logging/current_cost/ń;'Ew�+       ��K	6:���A�*

logging/current_cost�Ą;@���+       ��K	fl���A�*

logging/current_costzń;��y+       ��K	\����A�*

logging/current_cost�Ą;��J+       ��K	�֍��A�*

logging/current_cost�Ą;�)2A+       ��K	����A�*

logging/current_cost^Ą;�C�+       ��K	>���A�*

logging/current_cost�Ą;��F=+       ��K	�q���A�*

logging/current_costjƄ;����+       ��K	����A�*

logging/current_costń;�*W+       ��K	�ώ��A�*

logging/current_cost�Ą;_|�+       ��K	$����A�*

logging/current_costEń;��n�+       ��K	y1���A�*

logging/current_costqń;�T'}+       ��K	�_���A�*

logging/current_cost�Ą;�[+       ��K	䋏��A�*

logging/current_cost�Ä;���+       ��K	u����A�*

logging/current_cost�Ą;�J*�+       ��K	�珆�A�*

logging/current_cost[Ä;rO+       ��K	�7���A�*

logging/current_costGǄ;���+       ��K	ph���A�*

logging/current_cost4Ą;�?�_+       ��K	Ė���A�*

logging/current_cost�Ä;$T�r+       ��K	$Ð��A�*

logging/current_costZÄ;<
׫+       ��K	I��A�*

logging/current_cost�Ą;���e+       ��K	����A�*

logging/current_cost\Ą;��M+       ��K	�M���A�*

logging/current_cost�ń;��D�+       ��K	+|���A�*

logging/current_cost�Ä;u��y+       ��K	����A�*

logging/current_costCń;S�C�+       ��K	�ۑ��A�*

logging/current_cost�Ä;���<+       ��K	^���A�*

logging/current_costTĄ;�\N�+       ��K	:���A�*

logging/current_costLń;�g�+       ��K	gh���A�*

logging/current_cost�ń;Q�W+       ��K	t����A�*

logging/current_cost�;}�n�+       ��K	cĒ��A�*

logging/current_cost
Ȅ;9< �+       ��K	�����A�*

logging/current_costÄ;
�[k+       ��K	u)���A�*

logging/current_cost�;1�>+       ��K	�Y���A�*

logging/current_cost�;����+       ��K	g����A�*

logging/current_costsĄ;vDG+       ��K	Թ���A�*

logging/current_costWÄ;�Uځ+       ��K	�꓆�A�*

logging/current_cost�Ą;\$��+       ��K	����A�*

logging/current_cost���;�0?
+       ��K	�J���A�*

logging/current_costĄ;��G�+       ��K	�}���A�*

logging/current_costf;����+       ��K	*����A�*

logging/current_cost,Ä;=�L�+       ��K	8ٔ��A�*

logging/current_cost"Ą;�ڇ+       ��K	]���A�*

logging/current_cost�Ą;+�Z�+       ��K	�5���A�*

logging/current_costs��;M�+       ��K	g���A�*

logging/current_cost�Ƅ;��3%+       ��K	Ř���A�*

logging/current_cost���;Y��X+       ��K	wɕ��A�*

logging/current_costj��;��V+       ��K	�����A�*

logging/current_costY��;!{+       ��K	9+���A�*

logging/current_cost�;v�Gq+       ��K	E\���A�*

logging/current_cost;�ӆ�+       ��K	3����A�*

logging/current_cost-Ä;���+       ��K	�����A�*

logging/current_cost9��;�³+       ��K	�떆�A�*

logging/current_cost���;���*+       ��K	����A�*

logging/current_cost���;�.K�+       ��K	H���A�*

logging/current_cost.Ä;o��+       ��K	�{���A�*

logging/current_costZ��;9^�+       ��K	ɪ���A�*

logging/current_cost��;�f�;+       ��K	�ؗ��A�*

logging/current_cost9��;9�S+       ��K		���A�*

logging/current_cost�;�0'�+       ��K	�6���A�*

logging/current_cost�;��+       ��K	tg���A�*

logging/current_cost�;����+       ��K	&����A�*

logging/current_cost!��;9
��+       ��K		Ř��A�*

logging/current_cost俄;s��4+       ��K	�����A�*

logging/current_cost���;"�c�+       ��K	� ���A�*

logging/current_cost޿�;��S+       ��K	M���A�*

logging/current_cost*��;ͼ�K+       ��K	�|���A�*

logging/current_cost>��;��Lo+       ��K	�����A�*

logging/current_costf��;���+       ��K	�י��A�*

logging/current_costn��;��+       ��K	~���A�*

logging/current_cost��;1]5+       ��K	74���A�*

logging/current_cost���;&���+       ��K	V`���A�*

logging/current_cost��;Q���+       ��K	y����A�*

logging/current_cost���;���+       ��K	Ѽ���A�*

logging/current_cost��;U�NZ+       ��K	�뚆�A�*

logging/current_cost���;R���+       ��K	����A�*

logging/current_costi��;<mt�+       ��K	�F���A�*

logging/current_costU��;��+       ��K	�s���A�*

logging/current_costu��;g�d+       ��K	�����A�*

logging/current_costӽ�;� �+       ��K	�ϛ��A�*

logging/current_costh��;@8+       ��K	�����A�*

logging/current_cost���;fx��+       ��K	+���A�*

logging/current_cost]��;�FK�+       ��K	�W���A�*

logging/current_cost��;i}�8+       ��K	g����A�*

logging/current_cost���;��h�+       ��K	�����A�*

logging/current_costӽ�;st�+       ��K	�ܜ��A�*

logging/current_costu��;��Z�+       ��K	����A�*

logging/current_cost½�;��X�+       ��K	J:���A�*

logging/current_cost��;�d�T+       ��K	�f���A�*

logging/current_cost���;���#+       ��K	.����A�*

logging/current_costL��;�Z�+       ��K	9��A�*

logging/current_cost�;�Gn+       ��K	��A�*

logging/current_cost(��;6@TR+       ��K	����A�*

logging/current_costi��;�C�+       ��K	�L���A�*

logging/current_cost���;@��+       ��K	�y���A�*

logging/current_costk��;p�*U+       ��K	����A�*

logging/current_cost���;�[�+       ��K	ٞ��A�*

logging/current_cost���;��6+       ��K	���A�*

logging/current_cost޼�;~SP�+       ��K	5���A�*

logging/current_cost���;���+       ��K	�a���A�*

logging/current_costZ��;ok;+       ��K	1����A�*

logging/current_cost���;��l+       ��K	���A�*

logging/current_cost���;m���+       ��K	�����A�*

logging/current_costP��;��B	+       ��K	�'���A�*

logging/current_cost��;
Bi+       ��K	�U���A�*

logging/current_cost���;9��r+       ��K	.����A�*

logging/current_costH��;2S<�+       ��K	�����A�*

logging/current_cost¼�;e�bS+       ��K	$㠆�A�*

logging/current_costu��;�w9+       ��K	B���A�*

logging/current_cost���;���a+       ��K	�=���A�*

logging/current_cost<��;�m@+       ��K	�m���A�*

logging/current_cost���;��i
+       ��K	S����A�*

logging/current_cost��;R�P+       ��K	�ɡ��A�*

logging/current_cost��;�WP,+       ��K	�����A�*

logging/current_cost���;�y+       ��K	�*���A�*

logging/current_cost���;rw_�+       ��K	�[���A�*

logging/current_cost6��;<b+       ��K	�����A�*

logging/current_costغ�;�vP�+       ��K	0����A�*

logging/current_cost߻�;]gZ�+       ��K	좆�A�*

logging/current_costo��;�%�+       ��K	!���A�*

logging/current_cost���;��y+       ��K	G���A�*

logging/current_cost���;�D��+       ��K	uu���A�*

logging/current_costG��;��{+       ��K	̨���A�*

logging/current_costT��;��Y+       ��K	<֣��A�*

logging/current_cost���;a	+       ��K	����A�*

logging/current_costm��;�մ�+       ��K	�2���A�*

logging/current_costH��;֍�j+       ��K	Dg���A�*

logging/current_cost���;�nxF+       ��K	:����A�*

logging/current_costf��;)7 �+       ��K	�����A�*

logging/current_cost��;OU�!+       ��K	W��A�*

logging/current_cost���;���+       ��K	� ���A�*

logging/current_costi��;eX+       ��K	P���A�*

logging/current_cost���;G��+       ��K	���A�*

logging/current_costo��;Hz��+       ��K	Ѭ���A�*

logging/current_cost7��;�6g+       ��K	�ܥ��A�*

logging/current_cost���;9��+       ��K	%���A�*

logging/current_cost��;#�|�+       ��K	�:���A�*

logging/current_costC��;����+       ��K	i���A�*

logging/current_cost	��;NN��+       ��K	䖦��A�*

logging/current_cost׹�;5tL�+       ��K	cĦ��A�*

logging/current_cost���;�@#�+       ��K	M��A�*

logging/current_cost���;���+       ��K	�"���A�*

logging/current_cost���;^(7�+       ��K	&V���A�*

logging/current_cost�;l�C+       ��K	䄧��A�*

logging/current_cost%��;�7��+       ��K	ݲ���A�*

logging/current_costǸ�;��W�+       ��K	�৆�A�*

logging/current_cost5��;��+       ��K	"���A�*

logging/current_costS��;>�Y+       ��K	�>���A�*

logging/current_coste��;��+       ��K	l���A�*

logging/current_coste��;A�U+       ��K	�����A�*

logging/current_cost ��;����+       ��K	�ƨ��A�*

logging/current_cost?��;<��+       ��K	z��A�*

logging/current_cost��;�x�a+       ��K	I!���A�*

logging/current_costw��;�^(�+       ��K	�N���A�*

logging/current_cost���;���+       ��K	�|���A�*

logging/current_cost*��;���+       ��K	թ���A�*

logging/current_cost���;���$+       ��K	�ש��A�*

logging/current_cost¶�;�|/.+       ��K	����A�*

logging/current_costv��;��ƅ+       ��K	�3���A�*

logging/current_costm��;�o+       ��K	1a���A�*

logging/current_cost)��;]`�+       ��K	�����A�*

logging/current_cost-��;�@�+       ��K	��A�*

logging/current_cost���;"�_h+       ��K	�ꪆ�A�*

logging/current_cost\��;�T�5+       ��K	\���A�*

logging/current_costl��;�]ng+       ��K	oD���A�*

logging/current_costz��;�
h�+       ��K	\r���A�*

logging/current_cost'��;�\�+       ��K	ڠ���A�*

logging/current_cost���;���+       ��K	4Ы��A�*

logging/current_cost���;�l��+       ��K	 ���A�*

logging/current_costc��;�p�+       ��K	Z,���A�*

logging/current_cost��;�-��+       ��K	b[���A�*

logging/current_cost���;�Ja+       ��K	݈���A�*

logging/current_costD��;˟�+       ��K	�����A�*

logging/current_cost���;��L�+       ��K	�謆�A�*

logging/current_costǶ�; V�+       ��K	7���A�*

logging/current_costD��;^�P�+       ��K	`G���A�*

logging/current_cost��;��+       ��K	ft���A�*

logging/current_cost��;q��]+       ��K	�����A�*

logging/current_cost�;����+       ��K	8ҭ��A�*

logging/current_cost���;Jv�+       ��K	R ���A�*

logging/current_costQ��;��k +       ��K	�.���A�*

logging/current_cost[��;$���+       ��K	�^���A�*

logging/current_costR��;�%g+       ��K	;����A�*

logging/current_cost}��;���+       ��K	n����A�*

logging/current_cost&��;�ݶm+       ��K	�讆�A�*

logging/current_cost湄;�[�+       ��K	����A�*

logging/current_cost[��;2��+       ��K	TK���A�*

logging/current_cost~��;{C�)+       ��K	3x���A�*

logging/current_cost���;HB9�+       ��K	����A�*

logging/current_cost���;{$�+       ��K	`ԯ��A�*

logging/current_cost��;tА
+       ��K	����A�*

logging/current_cost��;U� S+       ��K	�5���A�*

logging/current_cost���;;(�+       ��K	2e���A�*

logging/current_cost;��;=Y�f+       ��K	ė���A�*

logging/current_costz��;�4��+       ��K	�Ű��A�*

logging/current_cost���;�0��+       ��K	n����A�*

logging/current_costζ�;�l�+       ��K	�$���A�*

logging/current_cost;���+       ��K	U���A�*

logging/current_costݴ�;��_m+       ��K	�����A�*

logging/current_cost��;Mt�+       ��K	񰱆�A� *

logging/current_costƳ�;���j+       ��K	�ޱ��A� *

logging/current_costⲄ;��+       ��K	����A� *

logging/current_costӴ�;O�&+       ��K	�?���A� *

logging/current_cost[��;�Zh�+       ��K	�m���A� *

logging/current_cost���;�%V�+       ��K	�����A� *

logging/current_costA��;}X+       ��K	~Ȳ��A� *

logging/current_cost���;�Ӑ+       ��K	+����A� *

logging/current_cost���;o�q�+       ��K	�'���A� *

logging/current_cost³�;'�+       ��K	$Y���A� *

logging/current_cost޳�;�i+       ��K	/����A� *

logging/current_cost���;�ůp+       ��K	����A� *

logging/current_cost���;�/.+       ��K	�ೆ�A� *

logging/current_cost��;��&+       ��K	,���A� *

logging/current_costU��;?���+       ��K	y?���A� *

logging/current_cost���;-:(�+       ��K	�n���A� *

logging/current_cost��;1��S+       ��K	����A� *

logging/current_cost	��;�n/+       ��K	�ʴ��A� *

logging/current_cost���;DP�C+       ��K	�����A� *

logging/current_cost;�L,+       ��K	�'���A� *

logging/current_costⲄ;�4�w+       ��K	XY���A� *

logging/current_costN��; �p{+       ��K	�����A� *

logging/current_costƳ�;� ��+       ��K	=����A� *

logging/current_cost��;��+       ��K	�㵆�A� *

logging/current_costZ��;/U�+       ��K	����A� *

logging/current_costϲ�;�E$)+       ��K	<>���A�!*

logging/current_cost�;wډ�+       ��K	jl���A�!*

logging/current_cost�;�T+       ��K	ś���A�!*

logging/current_cost$��;���s+       ��K	 ɶ��A�!*

logging/current_cost߲�;*o��+       ��K	�����A�!*

logging/current_costI��;��|+       ��K	r'���A�!*

logging/current_cost���;����+       ��K	3Y���A�!*

logging/current_cost鲄;���+       ��K	����A�!*

logging/current_cost���;v{!�+       ��K	Ҷ���A�!*

logging/current_costM��;Y	��+       ��K	z㷆�A�!*

logging/current_costy��;�k�+       ��K	����A�!*

logging/current_costN��;+HT+       ��K	�=���A�!*

logging/current_cost���;��+       ��K	�j���A�!*

logging/current_cost=��;g��+       ��K	����A�!*

logging/current_cost���;o=��+       ��K	pǸ��A�!*

logging/current_cost%��;���+       ��K	���A�!*

logging/current_costᰄ;��{&+       ��K	 ���A�!*

logging/current_cost ��;��Ĭ+       ��K	�N���A�!*

logging/current_cost{��;	�;�+       ��K	.}���A�!*

logging/current_costϰ�;��+       ��K	o����A�!*

logging/current_costU��;��W+       ��K	h׹��A�!*

logging/current_cost@��;d��[+       ��K	����A�!*

logging/current_cost���;��+       ��K	qB���A�!*

logging/current_costa��;��=+       ��K	3p���A�!*

logging/current_cost���;��=+       ��K	�����A�!*

logging/current_cost���;Y�$+       ��K	�κ��A�!*

logging/current_cost���;����+       ��K	m����A�"*

logging/current_cost���;�� o+       ��K	�,���A�"*

logging/current_costꮄ;M5�-+       ��K	�X���A�"*

logging/current_costͯ�;�ݓ�+       ��K	�����A�"*

logging/current_costư�;��J�+       ��K	'���A�"*

logging/current_costV��;fΎ�+       ��K	K���A�"*

logging/current_cost���;�e�S+       ��K	�����A�"*

logging/current_cost���;� "V+       ��K	C����A�"*

logging/current_costѯ�;o��+       ��K	��A�"*

logging/current_cost~��;�Θ+       ��K	����A�"*

logging/current_cost���;����+       ��K	BP���A�"*

logging/current_cost���;f.â+       ��K	~}���A�"*

logging/current_cost,��;DQ�+       ��K	s����A�"*

logging/current_cost���;(��+       ��K	8߽��A�"*

logging/current_cost���;���+       ��K	X���A�"*

logging/current_cost���;�K�+       ��K	?���A�"*

logging/current_cost߮�;tFQm+       ��K	1|���A�"*

logging/current_cost8��;D;��+       ��K	9����A�"*

logging/current_cost���;����+       ��K	�龆�A�"*

logging/current_costZ��;W��+       ��K		���A�"*

logging/current_costz��;Y�h�+       ��K	
M���A�"*

logging/current_costA��;��o+       ��K	d{���A�"*

logging/current_cost��;u�,+       ��K	�����A�"*

logging/current_costE��;���+       ��K	�ٿ��A�"*

logging/current_costV��;�d5X+       ��K	���A�"*

logging/current_cost}��;BO+       ��K	�:���A�#*

logging/current_cost���;�+�+       ��K	�i���A�#*

logging/current_cost���;��z+       ��K	����A�#*

logging/current_cost���;P�)�+       ��K	{����A�#*

logging/current_cost֬�;%�j�+       ��K	�����A�#*

logging/current_cost%��;�ă�+       ��K	�(���A�#*

logging/current_cost9��;����+       ��K	�[���A�#*

logging/current_costϬ�;���+       ��K	�����A�#*

logging/current_cost{��;zc��+       ��K	Z����A�#*

logging/current_cost��;�[�'+       ��K	�����A�#*

logging/current_costǮ�;�99+       ��K	p�A�#*

logging/current_costW��;�S~+       ��K	�E�A�#*

logging/current_cost��;o��+       ��K	�x�A�#*

logging/current_cost���;�n�+       ��K	q��A�#*

logging/current_cost|��;���+       ��K	r��A�#*

logging/current_cost���;�~�c+       ��K	Æ�A�#*

logging/current_costޭ�;���+       ��K	66Æ�A�#*

logging/current_cost���;c��,+       ��K	^eÆ�A�#*

logging/current_costૄ;�y��+       ��K	k�Æ�A�#*

logging/current_cost���;4c�-+       ��K	 �Æ�A�#*

logging/current_costԭ�;meO�+       ��K	��Æ�A�#*

logging/current_costŭ�;W>�u+       ��K	�/Ć�A�#*

logging/current_costR��;g#�+       ��K	�`Ć�A�#*

logging/current_costv��;��!+       ��K	��Ć�A�#*

logging/current_cost<��;�/0�+       ��K	G�Ć�A�#*

logging/current_coste��;(��+       ��K	��Ć�A�#*

logging/current_cost0��;ɍ�|+       ��K	X2ņ�A�$*

logging/current_cost���;oɥ�+       ��K	ueņ�A�$*

logging/current_cost���;��fS+       ��K	��ņ�A�$*

logging/current_cost�;��`�+       ��K	��ņ�A�$*

logging/current_costr��;���+       ��K	
Ɔ�A�$*

logging/current_cost���;�wj$+       ��K	�0Ɔ�A�$*

logging/current_cost���;�yLy+       ��K	DfƆ�A�$*

logging/current_cost���;�U1�+       ��K	͜Ɔ�A�$*

logging/current_costY��;T�k�+       ��K	d�Ɔ�A�$*

logging/current_cost骄;|^�+       ��K	hǆ�A�$*

logging/current_costY��;~p�|+       ��K	dKǆ�A�$*

logging/current_cost��;�3�+       ��K	��ǆ�A�$*

logging/current_cost���;�D+       ��K	
�ǆ�A�$*

logging/current_cost��;ȜJ+       ��K	��ǆ�A�$*

logging/current_cost���;�q�+       ��K	nȆ�A�$*

logging/current_cost;b��+       ��K	�MȆ�A�$*

logging/current_costܫ�;�@D�+       ��K	�Ȇ�A�$*

logging/current_cost���;g.��+       ��K	��Ȇ�A�$*

logging/current_cost䭄;��I+       ��K	B�Ȇ�A�$*

logging/current_costn��;c~��+       ��K	m,Ɇ�A�$*

logging/current_cost��;t^nZ+       ��K	�oɆ�A�$*

logging/current_cost˩�;E�w+       ��K	��Ɇ�A�$*

logging/current_cost{��;�f��+       ��K	(�Ɇ�A�$*

logging/current_cost���;��+       ��K	(ʆ�A�$*

logging/current_cost䫄;��Ka+       ��K	�Pʆ�A�$*

logging/current_cost��;��J�+       ��K	�ʆ�A�$*

logging/current_cost���;ċ�+       ��K	{�ʆ�A�%*

logging/current_costZ��;ߓe=+       ��K	�ʆ�A�%*

logging/current_cost4��;�y=�+       ��K	�1ˆ�A�%*

logging/current_cost߬�;Ӧ,�+       ��K	^dˆ�A�%*

logging/current_cost���;�v�+       ��K	�ˆ�A�%*

logging/current_cost���;sQ1+       ��K	#�ˆ�A�%*

logging/current_cost
��;+�w+       ��K	�̆�A�%*

logging/current_cost���;~�P+       ��K	�C̆�A�%*

logging/current_costͪ�;��+       ��K	�w̆�A�%*

logging/current_cost��;�.�+       ��K	
�̆�A�%*

logging/current_cost���;�?�+       ��K	Y�̆�A�%*

logging/current_cost���;�z�+       ��K	e&͆�A�%*

logging/current_costb��;���+       ��K	�\͆�A�%*

logging/current_cost���;����+       ��K	i�͆�A�%*

logging/current_cost�;��ض+       ��K	��͆�A�%*

logging/current_cost!��;*��B+       ��K	��͆�A�%*

logging/current_costb��;�q�+       ��K	 Ά�A�%*

logging/current_cost�;����+       ��K	RΆ�A�%*

logging/current_cost���;J��!+       ��K	r�Ά�A�%*

logging/current_cost���;�+&�+       ��K	��Ά�A�%*

logging/current_costI��;/u�n+       ��K	�φ�A�%*

logging/current_cost>��;�@�+       ��K	�6φ�A�%*

logging/current_cost5��;���+       ��K	�|φ�A�%*

logging/current_cost���;�� �+       ��K	�φ�A�%*

logging/current_cost٪�; �+       ��K	g�φ�A�%*

logging/current_cost`��;��+       ��K	P)І�A�&*

logging/current_costE��;b;$+       ��K	khІ�A�&*

logging/current_cost��;!��a+       ��K	�І�A�&*

logging/current_cost ��;���8+       ��K	n�І�A�&*

logging/current_cost���;�W��+       ��K	�>ц�A�&*

logging/current_cost���;l�q+       ��K	�xц�A�&*

logging/current_costh��;_u�#+       ��K	ıц�A�&*

logging/current_cost���;��+       ��K	��ц�A�&*

logging/current_cost멄;̚�2+       ��K	�#҆�A�&*

logging/current_costz��;y��+       ��K	CV҆�A�&*

logging/current_cost]��;�O;�+       ��K	)�҆�A�&*

logging/current_costت�;n)�+       ��K	)�҆�A�&*

logging/current_cost��;��+       ��K	��҆�A�&*

logging/current_costB��;[/�;+       ��K	�,ӆ�A�&*

logging/current_cost
��;�Ԏ+       ��K	o^ӆ�A�&*

logging/current_cost*��;� 0�+       ��K	G�ӆ�A�&*

logging/current_cost;A~Ҵ+       ��K	��ӆ�A�&*

logging/current_cost���;T\�+       ��K	�Ԇ�A�&*

logging/current_costN��;�߰�+       ��K	�AԆ�A�&*

logging/current_cost���;#��+       ��K		}Ԇ�A�&*

logging/current_cost���;��~+       ��K	ʭԆ�A�&*

logging/current_costg��;�&#+       ��K	A�Ԇ�A�&*

logging/current_cost妄;~��+       ��K	�Ն�A�&*

logging/current_cost{��;Zrf�+       ��K	=Ն�A�&*

logging/current_costS��;�l+       ��K	kՆ�A�&*

logging/current_costᨄ;\��!+       ��K	�Ն�A�&*

logging/current_cost���;����+       ��K	F�Ն�A�'*

logging/current_cost)��;.�Q+       ��K	��Ն�A�'*

logging/current_cost���;Պ��+       ��K	�&ֆ�A�'*

logging/current_cost���;`�̒+       ��K	�Uֆ�A�'*

logging/current_costT��;�a�+       ��K	ӎֆ�A�'*

logging/current_costۧ�;��+       ��K	f�ֆ�A�'*

logging/current_cost;`�lQ+       ��K	k׆�A�'*

logging/current_costǪ�;ҳ�`+       ��K	�E׆�A�'*

logging/current_cost��;�!�e+       ��K	�u׆�A�'*

logging/current_costŦ�;�㙉+       ��K	�׆�A�'*

logging/current_costK��;�� +       ��K	$�׆�A�'*

logging/current_cost)��;=��+       ��K	�؆�A�'*

logging/current_costH��;]���+       ��K	�6؆�A�'*

logging/current_cost���;��B+       ��K	�e؆�A�'*

logging/current_costF��;��!�+       ��K	!�؆�A�'*

logging/current_cost���;�i+       ��K	��؆�A�'*

logging/current_cost/��;���+       ��K	L�؆�A�'*

logging/current_costy��;���+       ��K	�#ن�A�'*

logging/current_costK��;���+       ��K	�Rن�A�'*

logging/current_costȦ�;����+       ��K	P�ن�A�'*

logging/current_cost���;��,#+       ��K	Ͱن�A�'*

logging/current_cost/��;�X�+       ��K	{�ن�A�'*

logging/current_costB��;�M�+       ��K	چ�A�'*

logging/current_costl��;��;�+       ��K	y@چ�A�'*

logging/current_cost宅;6�p+       ��K	�mچ�A�'*

logging/current_costۧ�;���I+       ��K	��چ�A�(*

logging/current_costߧ�;?���+       ��K	��چ�A�(*

logging/current_cost���;�F+       ��K	��چ�A�(*

logging/current_costb��;��c+       ��K	?.ۆ�A�(*

logging/current_cost���;���+       ��K	[ۆ�A�(*

logging/current_costO��;�-+       ��K	؆ۆ�A�(*

logging/current_cost���;@Z��+       ��K	?�ۆ�A�(*

logging/current_costӦ�;L��+       ��K	�ۆ�A�(*

logging/current_cost죄;�r�P+       ��K	�܆�A�(*

logging/current_cost���;?��+       ��K	�>܆�A�(*

logging/current_cost��;��*-+       ��K	�k܆�A�(*

logging/current_costx��;Ζ�t+       ��K	��܆�A�(*

logging/current_cost���;�0Z�+       ��K	 �܆�A�(*

logging/current_cost%��;��+       ��K	�݆�A�(*

logging/current_costҢ�;̽#l+       ��K	*4݆�A�(*

logging/current_cost͢�;ޢ�+       ��K	Ub݆�A�(*

logging/current_cost���;���+       ��K	n�݆�A�(*

logging/current_cost1��;���:+       ��K	�݆�A�(*

logging/current_cost#��;/�n+       ��K	�݆�A�(*

logging/current_cost���;���+       ��K	ކ�A�(*

logging/current_cost?��;~)�.+       ��K	�Lކ�A�(*

logging/current_cost�;���{+       ��K	Qyކ�A�(*

logging/current_cost٠�;:+��+       ��K	��ކ�A�(*

logging/current_costM��;p��$+       ��K	�ކ�A�(*

logging/current_cost龍;�Vi+       ��K	߆�A�(*

logging/current_cost���;M#JE+       ��K	s1߆�A�(*

logging/current_cost���;���+       ��K	�]߆�A�)*

logging/current_cost���;�JT+       ��K	��߆�A�)*

logging/current_cost��;{]\�+       ��K	̽߆�A�)*

logging/current_cost��;_���+       ��K	u�߆�A�)*

logging/current_costC��;�}��+       ��K	����A�)*

logging/current_costɠ�;�L�>+       ��K	�L���A�)*

logging/current_cost��;���+       ��K	i}���A�)*

logging/current_costɜ�;6��Z+       ��K	�����A�)*

logging/current_cost���;���Q+       ��K	$����A�)*

logging/current_cost���;}Z+       ��K	���A�)*

logging/current_cost���;/�s�+       ��K	�C��A�)*

logging/current_cost���;��x+       ��K	pu��A�)*

logging/current_cost��;!/��+       ��K	���A�)*

logging/current_cost���;)��K+       ��K	W���A�)*

logging/current_cost���;�G|+       ��K	���A�)*

logging/current_costx��;8c�/+       ��K	I+��A�)*

logging/current_cost̛�;/�q7+       ��K	4]��A�)*

logging/current_cost���;X���+       ��K	����A�)*

logging/current_costE��;���+       ��K	���A�)*

logging/current_cost�;���+       ��K	����A�)*

logging/current_cost���;�*��+       ��K	a��A�)*

logging/current_cost���;�>i�+       ��K	8>��A�)*

logging/current_cost���; ���+       ��K	dr��A�)*

logging/current_cost���;m���+       ��K	���A�)*

logging/current_cost���;ҀY+       ��K	����A�)*

logging/current_cost���;�5_y+       ��K	;���A�)*

logging/current_costa��;N5<+       ��K	�,��A�**

logging/current_cost���;���+       ��K	�^��A�**

logging/current_costy��;?/��+       ��K	����A�**

logging/current_cost���;uM� +       ��K	����A�**

logging/current_cost��;����+       ��K	����A�**

logging/current_costژ�;뼉`+       ��K	���A�**

logging/current_cost���;�N\%+       ��K	�G��A�**

logging/current_cost��;u�+       ��K	�t��A�**

logging/current_cost���;ɢ�I+       ��K	_���A�**

logging/current_cost9��;v
+       ��K	%���A�**

logging/current_cost
��;_�+       ��K	���A�**

logging/current_cost>��;do��+       ��K	�2��A�**

logging/current_cost~��;�+       ��K	�a��A�**

logging/current_cost���;9�i�+       ��K	����A�**

logging/current_costߘ�;��7+       ��K	���A�**

logging/current_costr��;�`�+       ��K	����A�**

logging/current_cost���;��+       ��K	��A�**

logging/current_cost���;Zw��+       ��K	P��A�**

logging/current_cost|��;��2+       ��K	`~��A�**

logging/current_cost���;�{�R+       ��K	����A�**

logging/current_costԘ�;CB�m+       ��K	����A�**

logging/current_costi��;ܷ�^+       ��K	���A�**

logging/current_cost��;��q+       ��K	�7��A�**

logging/current_costΕ�;�*�+       ��K	nk��A�**

logging/current_cost���;%@��+       ��K	����A�**

logging/current_cost��;�+	}+       ��K	z���A�+*

logging/current_cost0��;o,�+       ��K	 ���A�+*

logging/current_costE��;�\�+       ��K	�"��A�+*

logging/current_cost���;����+       ��K	�O��A�+*

logging/current_cost���;�6�+       ��K	���A�+*

logging/current_cost*��;g�q+       ��K	L���A�+*

logging/current_cost���;��e+       ��K	���A�+*

logging/current_cost���;�b�&+       ��K	f��A�+*

logging/current_cost���;�\�+       ��K	j4��A�+*

logging/current_cost��;�¥+       ��K	5`��A�+*

logging/current_cost%��;�W?�+       ��K		���A�+*

logging/current_cost���;n[HE+       ��K	���A�+*

logging/current_cost'��;�]H�+       ��K	����A�+*

logging/current_cost5��;	�+       ��K	)��A�+*

logging/current_cost현;݈+       ��K	�E��A�+*

logging/current_cost?��;4D_+       ��K	�s��A�+*

logging/current_cost'��;����+       ��K	����A�+*

logging/current_cost<��;�p��+       ��K	\���A�+*

logging/current_cost0��;��2{+       ��K	����A�+*

logging/current_cost���;�E�"+       ��K	�+��A�+*

logging/current_cost=��;�@2�+       ��K	PX��A�+*

logging/current_cost���;.�E�+       ��K	1���A�+*

logging/current_cost���;���+       ��K	����A�+*

logging/current_costO��;�/�/+       ��K	����A�+*

logging/current_cost)��;��t+       ��K	���A�+*

logging/current_cost]��; �f+       ��K	�<��A�+*

logging/current_cost���;��]+       ��K	j��A�,*

logging/current_cost���;,0�%+       ��K	 ���A�,*

logging/current_costĒ�;Tٕ;+       ��K	����A�,*

logging/current_cost���;z�3�+       ��K	���A�,*

logging/current_cost'��;;$-�+       ��K		"��A�,*

logging/current_cost,��;Z��+       ��K	�Z��A�,*

logging/current_costF��;1���+       ��K	\���A�,*

logging/current_cost�;�C�i+       ��K	3���A�,*

logging/current_cost���;�� �+       ��K	����A�,*

logging/current_cost���;�Ǿ�+       ��K	���A�,*

logging/current_costg��;��+       ��K	oA��A�,*

logging/current_cost��;�p�@+       ��K	�o��A�,*

logging/current_cost���;gI��+       ��K	'���A�,*

logging/current_cost~��;�!�w+       ��K	|���A�,*

logging/current_costf��;K�J�+       ��K	����A�,*

logging/current_cost~��;��-�+       ��K	\-���A�,*

logging/current_costa��;P�+       ��K	BZ���A�,*

logging/current_costc��;T}E}+       ��K	͈���A�,*

logging/current_costm��;�t=�+       ��K	����A�,*

logging/current_cost���;����+       ��K	�����A�,*

logging/current_cost-��;9��+       ��K	d��A�,*

logging/current_cost~��;��w�+       ��K	pG��A�,*

logging/current_cost$��;�s�+       ��K	�v��A�,*

logging/current_cost<��;�+       ��K	N���A�,*

logging/current_cost���;&a�+       ��K	}���A�,*

logging/current_cost���;�z]�+       ��K	��A�-*

logging/current_cost>��;8V��+       ��K	!1��A�-*

logging/current_cost(��;8lm�+       ��K	X`��A�-*

logging/current_cost`��;.��+       ��K	H���A�-*

logging/current_cost���;i�+       ��K	q���A�-*

logging/current_cost�;+k��+       ��K	����A�-*

logging/current_cost���;X��%+       ��K	���A�-*

logging/current_cost*��;�%�+       ��K	HH��A�-*

logging/current_cost+��;\��`+       ��K	��A�-*

logging/current_cost���;�$�+       ��K	����A�-*

logging/current_cost���;�� )+       ��K	\���A�-*

logging/current_cost蓄;�y~�+       ��K	C��A�-*

logging/current_cost���;��}+       ��K	_9��A�-*

logging/current_cost ��;|�y+       ��K	Aj��A�-*

logging/current_cost���;����+       ��K	����A�-*

logging/current_costA��;%�� +       ��K		���A�-*

logging/current_cost���;�w�+       ��K	���A�-*

logging/current_cost���;�c�+       ��K	� ���A�-*

logging/current_cost��;��i8+       ��K	Q���A�-*

logging/current_cost���;�9�+       ��K	y����A�-*

logging/current_costG��;��4�+       ��K	Q����A�-*

logging/current_cost���;��+       ��K	����A�-*

logging/current_cost鑄;u[�l+       ��K	���A�-*

logging/current_costԑ�;�*��+       ��K	AJ���A�-*

logging/current_cost���;Y���+       ��K	�~���A�-*

logging/current_cost���;�Q�+       ��K	1����A�-*

logging/current_costZ��;ݜ��+       ��K	�����A�.*

logging/current_cost���;�e�+       ��K	#���A�.*

logging/current_costz��;���+       ��K	�I���A�.*

logging/current_cost���;��t+       ��K	Y���A�.*

logging/current_costА�;�0��+       ��K	�����A�.*

logging/current_cost ��; �v+       ��K	�����A�.*

logging/current_costЏ�;�泿+       ��K	���A�.*

logging/current_cost"��;qƁ+       ��K	L@���A�.*

logging/current_cost���; *��+       ��K	 p���A�.*

logging/current_costu��;?��+       ��K	\����A�.*

logging/current_cost��;�-+       ��K	
����A�.*

logging/current_cost5��;�.+       ��K	����A�.*

logging/current_cost��;��)�+       ��K	�.���A�.*

logging/current_cost���;F���+       ��K	�[���A�.*

logging/current_cost͑�;|��?+       ��K	g����A�.*

logging/current_cost,��;O�E~+       ��K	�����A�.*

logging/current_cost��;�!+       ��K	3����A�.*

logging/current_cost���;Q+7�+       ��K	���A�.*

logging/current_costt��;V�K+       ��K	�L���A�.*

logging/current_costC��;٦�+       ��K	z{���A�.*

logging/current_cost���;OcT+       ��K	w����A�.*

logging/current_costŏ�;R�n+       ��K	�����A�.*

logging/current_cost���;͹��+       ��K	����A�.*

logging/current_cost���;,�+       ��K	�9���A�.*

logging/current_cost���;�=d+       ��K	�i���A�.*

logging/current_cost���;CW�'+       ��K	�����A�.*

logging/current_costd��;/�T�+       ��K	=���A�/*

logging/current_cost���;	lU�+       ��K	�T���A�/*

logging/current_costU��;�O�+       ��K	<����A�/*

logging/current_cost^��;��b+       ��K	|����A�/*

logging/current_cost���;���+       ��K	h���A�/*

logging/current_costq��;*"Ko+       ��K	mH���A�/*

logging/current_costՎ�;]�o�+       ��K	n����A�/*

logging/current_cost���;���+       ��K	b����A�/*

logging/current_costY��;#B��+       ��K	�����A�/*

logging/current_costN��;E��+       ��K		'���A�/*

logging/current_cost��;��&+       ��K	�j���A�/*

logging/current_cost���;>�JT+       ��K	͠���A�/*

logging/current_cost��;�:q�+       ��K	�����A�/*

logging/current_cost+��;���T+       ��K	���A�/*

logging/current_cost���;�)�\+       ��K	@���A�/*

logging/current_costq��;d�44+       ��K	�u���A�/*

logging/current_cost;�54e+       ��K	զ���A�/*

logging/current_cost���;5M�K+       ��K	�����A�/*

logging/current_costu��;FxEJ+       ��K	� ��A�/*

logging/current_costy��;w6r�+       ��K	�T ��A�/*

logging/current_cost���;���N+       ��K	�� ��A�/*

logging/current_costޔ�;G"�+       ��K	�� ��A�/*

logging/current_costF��;$�+       ��K	���A�/*

logging/current_costז�;%6�#+       ��K	$I��A�/*

logging/current_cost���;�lN�+       ��K	����A�/*

logging/current_costv��;~ӵ+       ��K	ǻ��A�0*

logging/current_cost菄;b�O+       ��K	��A�0*

logging/current_cost���;fSV+       ��K	Q���A�0*

logging/current_cost��;AH�f+       ��K	k���A�0*

logging/current_costv��;�H9+       ��K	d��A�0*

logging/current_costd��;L�]+       ��K	w`��A�0*

logging/current_costJ��;2�k�+       ��K	����A�0*

logging/current_cost���;���+       ��K	����A�0*

logging/current_cost��;&OH�+       ��K	*��A�0*

logging/current_cost/��;m��+       ��K	�T��A�0*

logging/current_cost���;�y�}+       ��K	/���A�0*

logging/current_costG��;���+       ��K	c���A�0*

logging/current_cost���;_\��+       ��K	_ ��A�0*

logging/current_cost̑�;$��+       ��K	�4��A�0*

logging/current_costH��;��&+       ��K	�d��A�0*

logging/current_costs��;O�ů+       ��K	����A�0*

logging/current_cost���;$��{+       ��K	?���A�0*

logging/current_cost��;���+       ��K	z��A�0*

logging/current_cost���;u��+       ��K	�<��A�0*

logging/current_costD��;-�r^+       ��K	�v��A�0*

logging/current_costP��;�
+       ��K	@���A�0*

logging/current_costؒ�;�)�+       ��K	����A�0*

logging/current_costn��;]�%�+       ��K	�6��A�0*

logging/current_costb��;�Q�+       ��K	,o��A�0*

logging/current_costꎄ;�o�k+       ��K	"���A�0*

logging/current_costy��;ܯˤ+       ��K	q���A�0*

logging/current_cost��;ɡ�%+       ��K	���A�1*

logging/current_cost̏�;c�%+       ��K	RU��A�1*

logging/current_cost���;P0�+       ��K	<���A�1*

logging/current_cost���;:Y0�+       ��K	����A�1*

logging/current_costt��;W��k+       ��K		��A�1*

logging/current_cost(��;|�y+       ��K	�D	��A�1*

logging/current_cost���;���+       ��K	Y�	��A�1*

logging/current_cost͍�;C��+       ��K	��	��A�1*

logging/current_costf��;1<`�+       ��K	�
��A�1*

logging/current_cost鎄;��+       ��K		W
��A�1*

logging/current_costZ��;ߝ��+       ��K	��
��A�1*

logging/current_coste��;�%+       ��K	�
��A�1*

logging/current_cost���;�EG�+       ��K	���A�1*

logging/current_cost���;�6�+       ��K	L6��A�1*

logging/current_costz��;�B�+       ��K	�p��A�1*

logging/current_cost���;�ǰJ+       ��K	����A�1*

logging/current_costO��;�K`+       ��K	����A�1*

logging/current_cost���;&7��+       ��K	�.��A�1*

logging/current_costs��;���+       ��K	zf��A�1*

logging/current_costu��;�9zT+       ��K	%���A�1*

logging/current_costU��;����+       ��K	x���A�1*

logging/current_costR��;5(p�+       ��K	P��A�1*

logging/current_costG��;chر+       ��K	;<��A�1*

logging/current_costk��;!���+       ��K	!m��A�1*

logging/current_cost���;=�D�+       ��K	W���A�1*

logging/current_cost���;?s!++       ��K	F���A�2*

logging/current_cost���;��+       ��K	���A�2*

logging/current_costg��;���+       ��K	�R��A�2*

logging/current_costA��;���+       ��K	%���A�2*

logging/current_cost��;j�9�+       ��K	����A�2*

logging/current_cost���;	�O�+       ��K	����A�2*

logging/current_costq��;�� +       ��K	� ��A�2*

logging/current_cost���;(�In+       ��K	tU��A�2*

logging/current_cost%��;	��e+       ��K	L���A�2*

logging/current_costL��;�-��+       ��K	Խ��A�2*

logging/current_costr��;ַ�#+       ��K	���A�2*

logging/current_cost;��u�+       ��K	�7��A�2*

logging/current_cost���;� �d+       ��K	����A�2*

logging/current_cost���;���<+       ��K	=���A�2*

logging/current_cost��;Q'��+       ��K	H���A�2*

logging/current_cost���;5�5�+       ��K	���A�2*

logging/current_cost���;lܰ^+       ��K	^M��A�2*

logging/current_cost���;���+       ��K	:���A�2*

logging/current_cost^��;]]�+       ��K	õ��A�2*

logging/current_cost���;���+       ��K	����A�2*

logging/current_cost��;e��$+       ��K	��A�2*

logging/current_cost׍�;���:+       ��K	PM��A�2*

logging/current_cost<��;l+       ��K	����A�2*

logging/current_cost���;V��B+       ��K	Ҷ��A�2*

logging/current_costʐ�;�\j�+       ��K	R���A�2*

logging/current_costj��;�gU�+       ��K	0��A�2*

logging/current_cost���;�3�+       ��K	mP��A�3*

logging/current_cost1��;��b�+       ��K	g���A�3*

logging/current_cost���;�cs+       ��K	h���A�3*

logging/current_cost���;F(�*+       ��K	����A�3*

logging/current_cost��;���+       ��K	*��A�3*

logging/current_cost���;�f�+       ��K	e��A�3*

logging/current_costƎ�;It��+       ��K	e���A�3*

logging/current_cost>��;vm�{+       ��K	a���A�3*

logging/current_cost{��;��k+       ��K	���A�3*

logging/current_cost���;lq�e+       ��K	T?��A�3*

logging/current_cost���;��!�+       ��K	�|��A�3*

logging/current_cost?��;�k�+       ��K	x���A�3*

logging/current_cost}��; \�+       ��K	(���A�3*

logging/current_cost���;F�+       ��K	�+��A�3*

logging/current_cost��;v�+       ��K	ag��A�3*

logging/current_costዄ;BMX�+       ��K	����A�3*

logging/current_cost���;��s+       ��K	,���A�3*

logging/current_costዄ;��٬+       ��K	_*��A�3*

logging/current_cost���;�JQx+       ��K	�b��A�3*

logging/current_cost��;�ii+       ��K	P���A�3*

logging/current_cost���;U	+       ��K	����A�3*

logging/current_cost���;�:�+       ��K	���A�3*

logging/current_costy��;�	D�+       ��K	hY��A�3*

logging/current_costY��;FTo+       ��K	]���A�3*

logging/current_costፄ;�J� +       ��K	����A�3*

logging/current_cost`��;�B-z+       ��K	*��A�3*

logging/current_costL��;&~+       ��K	�M��A�4*

logging/current_cost�;���h+       ��K	���A�4*

logging/current_costb��;6��*+       ��K	}���A�4*

logging/current_cost���;E�a+       ��K	U���A�4*

logging/current_cost���;'(�+       ��K	�)��A�4*

logging/current_cost���;I\%�+       ��K	r[��A�4*

logging/current_cost!��;����+       ��K	����A�4*

logging/current_cost���;��(U+       ��K	���A�4*

logging/current_cost>��;��i�+       ��K	���A�4*

logging/current_costp��;e�*�+       ��K	kB��A�4*

logging/current_cost���;��S�+       ��K	���A�4*

logging/current_cost$��;���f+       ��K	���A�4*

logging/current_cost���;*"��+       ��K	F��A�4*

logging/current_cost���;��+       ��K	qy��A�4*

logging/current_cost��;��+       ��K	���A�4*

logging/current_cost'��;���+       ��K	����A�4*

logging/current_cost֋�;X,�+       ��K	+*��A�4*

logging/current_costg��;Lq0+       ��K	�]��A�4*

logging/current_cost��;	?<+       ��K	����A�4*

logging/current_cost.��;Ll;)+       ��K	P���A�4*

logging/current_cost䏄;adS�+       ��K	���A�4*

logging/current_cost ��;
x�'+       ��K	xG��A�4*

logging/current_cost��;��Q+       ��K	S~��A�4*

logging/current_cost���;��Z�+       ��K	���A�4*

logging/current_cost���;��+       ��K	���A�4*

logging/current_cost
��;o?�+       ��K	�'��A�5*

logging/current_cost���;��+       ��K	�l��A�5*

logging/current_cost%��;�5Ʒ+       ��K	���A�5*

logging/current_costÑ�;�J+       ��K	���A�5*

logging/current_cost��;$�B+       ��K	m4 ��A�5*

logging/current_cost���;s%HS+       ��K	*j ��A�5*

logging/current_cost���;jE|k+       ��K	h� ��A�5*

logging/current_cost�;�c�1+       ��K	R� ��A�5*

logging/current_cost��;���+       ��K	�!��A�5*

logging/current_cost��;;_9�+       ��K	�Q!��A�5*

logging/current_cost���;8��+       ��K	��!��A�5*

logging/current_costB��;��`+       ��K	޽!��A�5*

logging/current_cost���;�u��+       ��K	��!��A�5*

logging/current_cost/��;3mRQ+       ��K	&"��A�5*

logging/current_cost_��;�+       ��K	@d"��A�5*

logging/current_cost싄;Gm�+       ��K	њ"��A�5*

logging/current_costd��;�gOl+       ��K	��"��A�5*

logging/current_costt��;����+       ��K	��"��A�5*

logging/current_costQ��;ڨO+       ��K	9#��A�5*

logging/current_cost�;���+       ��K	$v#��A�5*

logging/current_cost1��;��N�+       ��K	S�#��A�5*

logging/current_costb��; +��+       ��K	�#��A�5*

logging/current_cost���;�/��+       ��K	N$��A�5*

logging/current_cost���;��+       ��K	aK$��A�5*

logging/current_cost��;!+u+       ��K	��$��A�5*

logging/current_cost���;}P�]+       ��K	O�$��A�5*

logging/current_cost���;��)+       ��K	��$��A�6*

logging/current_cost��;{Zv�+       ��K	*(%��A�6*

logging/current_costΊ�;��8{+       ��K	Jc%��A�6*

logging/current_cost���;��v+       ��K	h�%��A�6*

logging/current_cost���;0�+       ��K	��%��A�6*

logging/current_cost���;�͝+       ��K	�&��A�6*

logging/current_cost鏄;��+       ��K	C8&��A�6*

logging/current_cost���;
:��+       ��K	-m&��A�6*

logging/current_costb��;4�;~+       ��K	N�&��A�6*

logging/current_costu��;�8s�+       ��K	\�&��A�6*

logging/current_costs��;�Go�+       ��K	'��A�6*

logging/current_cost׌�;2Vi�+       ��K	gS'��A�6*

logging/current_costj��;��a+       ��K	L�'��A�6*

logging/current_cost͊�;�<0�+       ��K	��'��A�6*

logging/current_costJ��;��{�+       ��K	Q(��A�6*

logging/current_cost���;@[/d+       ��K	K(��A�6*

logging/current_costی�;ۮ�?+       ��K	�(��A�6*

logging/current_cost��;+�AA+       ��K	�(��A�6*

logging/current_cost��;�XK�+       ��K	%�(��A�6*

logging/current_cost��;JR�8+       ��K	%7)��A�6*

logging/current_cost6��;���+       ��K	@�)��A�6*

logging/current_cost틄;����+       ��K	�)��A�6*

logging/current_cost`��;;�+       ��K	*��A�6*

logging/current_cost���;�Y�+       ��K	�L*��A�6*

logging/current_costא�;Ki�+       ��K	��*��A�6*

logging/current_cost���;ac�v+       ��K	6�*��A�7*

logging/current_cost���;��3+       ��K	�+��A�7*

logging/current_cost%��;�q��+       ��K	�E+��A�7*

logging/current_costp��;����+       ��K	�u+��A�7*

logging/current_cost���;�b%r+       ��K	�+��A�7*

logging/current_costb��;�Q/�+       ��K	��+��A�7*

logging/current_cost��;�tRK+       ��K	,��A�7*

logging/current_cost��;N>�P+       ��K	�P,��A�7*

logging/current_costb��;/?+       ��K	N�,��A�7*

logging/current_cost狄;���+       ��K	�,��A�7*

logging/current_cost���;��dT+       ��K	t�,��A�7*

logging/current_cost֊�;χYS+       ��K	$--��A�7*

logging/current_cost&��;���+       ��K	>`-��A�7*

logging/current_cost��;U�Z�+       ��K	��-��A�7*

logging/current_cost鍄;2Q�+       ��K	7�-��A�7*

logging/current_costǎ�;���+       ��K	'.��A�7*

logging/current_costኄ;���}+       ��K	N.��A�7*

logging/current_costi��;�5�+       ��K	��.��A�7*

logging/current_cost ��;%�&/+       ��K	;�.��A�7*

logging/current_cost���;_�ď+       ��K	S/��A�7*

logging/current_coste��;ɒ�M+       ��K	�_/��A�7*

logging/current_cost��;c\�+       ��K	�/��A�7*

logging/current_cost ��;U�/�+       ��K	��/��A�7*

logging/current_costڎ�;�t�+       ��K	��/��A�7*

logging/current_cost;`@,+       ��K	;=0��A�7*

logging/current_cost���;�~��+       ��K	�t0��A�7*

logging/current_cost싄;5��!+       ��K	ݼ0��A�8*

logging/current_cost��;�*��+       ��K	��0��A�8*

logging/current_cost���;�-w�+       ��K	�81��A�8*

logging/current_cost���;�h�+       ��K	�q1��A�8*

logging/current_cost��;�z�+       ��K	b�1��A�8*

logging/current_costӍ�;!�A+       ��K	�1��A�8*

logging/current_cost݌�;[L�"+       ��K	$2��A�8*

logging/current_cost���;ѐ��+       ��K	`E2��A�8*

logging/current_cost���;B��+       ��K	�2��A�8*

logging/current_cost��;����+       ��K	�2��A�8*

logging/current_cost��;�&+       ��K	Z�2��A�8*

logging/current_costz��;���+       ��K	>3��A�8*

logging/current_costȋ�;��>#+       ��K	Dy3��A�8*

logging/current_cost���;.	q�+       ��K	�3��A�8*

logging/current_cost��;J�C�+       ��K	F�3��A�8*

logging/current_cost�;UZ��+       ��K	4��A�8*

logging/current_cost茄;��+       ��K	(Q4��A�8*

logging/current_cost\��;d���+       ��K	n�4��A�8*

logging/current_costK��;���+       ��K	0�4��A�8*

logging/current_cost)��;7�&~+       ��K	+5��A�8*

logging/current_costk��;q�+       ��K	OJ5��A�8*

logging/current_cost9��;|�	�+       ��K	5��A�8*

logging/current_costb��;���+       ��K	�5��A�8*

logging/current_costd��;����+       ��K	�5��A�8*

logging/current_costR��;q�wO+       ��K	�6��A�8*

logging/current_cost��;�(��+       ��K	�I6��A�8*

logging/current_cost芄;�U�+       ��K	�6��A�9*

logging/current_cost���;;jժ+       ��K	O�6��A�9*

logging/current_cost܉�;�lV+       ��K	I7��A�9*

logging/current_cost���;@�@�+       ��K	c<7��A�9*

logging/current_cost?��;z��m+       ��K	+|7��A�9*

logging/current_cost抄;*ў+       ��K	�7��A�9*

logging/current_costf��;�l+       ��K	��7��A�9*

logging/current_cost���;�{R�+       ��K	# 8��A�9*

logging/current_cost���;q�OH+       ��K	BP8��A�9*

logging/current_costƊ�;q#�+       ��K	:�8��A�9*

logging/current_cost���;*���+       ��K	��8��A�9*

logging/current_costv��;�=+       ��K	�8��A�9*

logging/current_cost덄;�T64+       ��K	� 9��A�9*

logging/current_cost���;:�;�+       ��K	&N9��A�9*

logging/current_cost썄;��=+       ��K	(9��A�9*

logging/current_cost��;\�^6+       ��K	�9��A�9*

logging/current_cost���;��5N+       ��K	��9��A�9*

logging/current_cost���;�6�g+       ��K	� :��A�9*

logging/current_cost���;�+��+       ��K	�Z:��A�9*

logging/current_costK��;���+       ��K	��:��A�9*

logging/current_cost���;q �&+       ��K	��:��A�9*

logging/current_costϊ�;3�c	+       ��K	4;��A�9*

logging/current_cost싄;��+       ��K	pG;��A�9*

logging/current_costҋ�;'��1+       ��K	O�;��A�9*

logging/current_cost_��;�⫄+       ��K	l�;��A�9*

logging/current_cost���;̖%+       ��K	"<��A�:*

logging/current_cost���;�r�+       ��K	 `<��A�:*

logging/current_cost���;�Q�j+       ��K	��<��A�:*

logging/current_cost��;W��k+       ��K	�<��A�:*

logging/current_cost׊�;ϴV�+       ��K	o#=��A�:*

logging/current_cost���;��8+       ��K	=V=��A�:*

logging/current_cost��;���+       ��K	͜=��A�:*

logging/current_cost���;y��l+       ��K	�=��A�:*

logging/current_costM��;��U�+       ��K	�>��A�:*

logging/current_cost���;jr�+       ��K	�W>��A�:*

logging/current_cost؎�;}�u�+       ��K	G�>��A�:*

logging/current_cost��;V@�+       ��K	��>��A�:*

logging/current_cost���;%n�+       ��K	�>��A�:*

logging/current_cost�;Tc:,+       ��K	�'?��A�:*

logging/current_cost���;�^~+       ��K	�b?��A�:*

logging/current_cost抄;�1�+       ��K	w�?��A�:*

logging/current_cost���;0�ϴ+       ��K	+�?��A�:*

logging/current_costL��;��+       ��K	6@��A�:*

logging/current_cost֋�;r��+       ��K	�I@��A�:*

logging/current_costY��;���C+       ��K	�@��A�:*

logging/current_cost ��;��E+       ��K	�@��A�:*

logging/current_cost���;N��+       ��K	-	A��A�:*

logging/current_cost���;Pg+       ��K	uCA��A�:*

logging/current_cost��;�&�+       ��K	�yA��A�:*

logging/current_cost���;���+       ��K	U�A��A�:*

logging/current_costr��;�T~+       ��K	��A��A�:*

logging/current_costǌ�;�!��+       ��K	�B��A�;*

logging/current_cost6��;�/�+       ��K	�VB��A�;*

logging/current_cost���;��w�+       ��K	7�B��A�;*

logging/current_cost���;����+       ��K	_�B��A�;*

logging/current_cost���;pae�+       ��K	PC��A�;*

logging/current_cost���; -��+       ��K	��C��A�;*

logging/current_cost���;X�w+       ��K	P�C��A�;*

logging/current_cost��;sRAH+       ��K	�XD��A�;*

logging/current_costf��; 7��+       ��K	N�D��A�;*

logging/current_cost܏�;�`�+       ��K	�E��A�;*

logging/current_costS��;J_��+       ��K	:VE��A�;*

logging/current_cost���;����+       ��K	��E��A�;*

logging/current_cost���;���+       ��K	A�E��A�;*

logging/current_cost���;r��+       ��K	CF��A�;*

logging/current_cost~��;��L�+       ��K	 >F��A�;*

logging/current_cost���;D�K>+       ��K	 sF��A�;*

logging/current_cost���;���+       ��K	ϥF��A�;*

logging/current_cost���;���+       ��K	,�F��A�;*

logging/current_costq��;����+       ��K	\G��A�;*

logging/current_cost썄;��T�+       ��K	|PG��A�;*

logging/current_cost��;�q35+       ��K	n�G��A�;*

logging/current_cost��;���0+       ��K	��G��A�;*

logging/current_costՎ�;+�*�+       ��K	�H��A�;*

logging/current_cost���;����+       ��K	EH��A�;*

logging/current_cost͌�;/�z+       ��K	yH��A�;*

logging/current_costS��;����+       ��K	��H��A�<*

logging/current_cost���;�W�9+       ��K	�I��A�<*

logging/current_cost��;rpAC+       ��K	�II��A�<*

logging/current_costr��;��+       ��K	�}I��A�<*

logging/current_costۉ�;�D��+       ��K	,�I��A�<*

logging/current_costǌ�;���+       ��K	�J��A�<*

logging/current_cost���;9��u+       ��K	e\J��A�<*

logging/current_costK��;�7P�+       ��K	��J��A�<*

logging/current_costފ�;˩K�+       ��K	n�J��A�<*

logging/current_coste��;'�u+       ��K	��J��A�<*

logging/current_costG��;��5�+       ��K	J0K��A�<*

logging/current_costS��;C�F�+       ��K	�bK��A�<*

logging/current_costt��;86l+       ��K	��K��A�<*

logging/current_cost���;rK�+       ��K	�K��A�<*

logging/current_cost��;v]+       ��K	+L��A�<*

logging/current_cost׊�;LQ�s+       ��K	GL��A�<*

logging/current_cost���;"�+       ��K	��L��A�<*

logging/current_cost���;�Au+       ��K	�L��A�<*

logging/current_cost��;�D5+       ��K	HM��A�<*

logging/current_cost���;%�+       ��K	��M��A�<*

logging/current_costC��;��:+       ��K	��M��A�<*

logging/current_cost���;b���+       ��K	��M��A�<*

logging/current_cost���;���/+       ��K	R#N��A�<*

logging/current_cost䌄;-��+       ��K	�YN��A�<*

logging/current_costB��;F��+       ��K	��N��A�<*

logging/current_cost��;��x+       ��K	�N��A�<*

logging/current_cost���;ꌍe+       ��K	n�N��A�=*

logging/current_cost���;���+       ��K	c3O��A�=*

logging/current_costĊ�;���+       ��K	lO��A�=*

logging/current_cost���;h�q�+       ��K	:�O��A�=*

logging/current_costӌ�;oz
U+       ��K	*�O��A�=*

logging/current_cost^��;^�N�+       ��K	MP��A�=*

logging/current_cost��;Jx&:+       ��K	aAP��A�=*

logging/current_cost%��;����+       ��K	�vP��A�=*

logging/current_costS��;���g+       ��K	��P��A�=*

logging/current_costQ��;�<��+       ��K	Y�P��A�=*

logging/current_cost���;�L%�+       ��K	WQ��A�=*

logging/current_cost5��;�b��+       ��K	�@Q��A�=*

logging/current_costc��;BH�+       ��K	EyQ��A�=*

logging/current_cost��;٭��+       ��K	@�Q��A�=*

logging/current_costË�;�0�+       ��K	��Q��A�=*

logging/current_cost���;���M+       ��K	wR��A�=*

logging/current_costl��;�@�[+       ��K	�QR��A�=*

logging/current_cost݉�;���>+       ��K	��R��A�=*

logging/current_cost/��;���u+       ��K	[�R��A�=*

logging/current_cost���;BG+       ��K	MS��A�=*

logging/current_cost&��;����+       ��K	�US��A�=*

logging/current_cost���;�Iڍ+       ��K	]�S��A�=*

logging/current_cost��;��mb+       ��K	˿S��A�=*

logging/current_cost4��;���A+       ��K	
T��A�=*

logging/current_cost���;O�a:+       ��K	�FT��A�=*

logging/current_cost�;^$�+       ��K	<�T��A�=*

logging/current_costH��;�+       ��K	��T��A�>*

logging/current_costf��;,�`z+       ��K	��T��A�>*

logging/current_costO��;]ƪ�+       ��K	�/U��A�>*

logging/current_cost��;�B +       ��K	�lU��A�>*

logging/current_cost9��;#6�+       ��K	��U��A�>*

logging/current_cost���;���h+       ��K	��U��A�>*

logging/current_cost8��;��(�+       ��K	XV��A�>*

logging/current_cost���;�3~+       ��K	V>V��A�>*

logging/current_cost8��;Z��d+       ��K	'V��A�>*

logging/current_costߌ�;R|4!+       ��K	h�V��A�>*

logging/current_cost��;:o�+       ��K	��V��A�>*

logging/current_cost���;Hk\'+       ��K	�&W��A�>*

logging/current_costƊ�;�i�%+       ��K	�ZW��A�>*

logging/current_cost=��;�B,+       ��K	,�W��A�>*

logging/current_costx��;��" +       ��K	��W��A�>*

logging/current_cost���;�`R+       ��K	jX��A�>*

logging/current_costI��;#}p�+       ��K	QHX��A�>*

logging/current_cost���;!m,+       ��K	��X��A�>*

logging/current_costO��;�W�+       ��K	{�X��A�>*

logging/current_cost̉�;T�+       ��K	�/Y��A�>*

logging/current_cost닄;�:P�+       ��K	�gY��A�>*

logging/current_cost|��;�H{+       ��K	p�Y��A�>*

logging/current_cost��;���+       ��K	 �Y��A�>*

logging/current_costP��;Y�4+       ��K	zZ��A�>*

logging/current_cost���;�s+       ��K	bIZ��A�>*

logging/current_costӋ�;���+       ��K	!�Z��A�?*

logging/current_cost���;m{y�+       ��K	��Z��A�?*

logging/current_costڊ�;Иs�+       ��K	�Z��A�?*

logging/current_cost��;��i3+       ��K	�>[��A�?*

logging/current_cost���;�H+       ��K	�t[��A�?*

logging/current_cost$��;@�tO+       ��K	��[��A�?*

logging/current_cost ��;I�J�+       ��K	��[��A�?*

logging/current_costf��;0��R+       ��K	�\��A�?*

logging/current_cost���;��+       ��K	�L\��A�?*

logging/current_costW��;dA<�+       ��K	ف\��A�?*

logging/current_costS��;��+       ��K	��\��A�?*

logging/current_cost���;�`O�+       ��K	�\��A�?*

logging/current_cost��;ơO+       ��K	b5]��A�?*

logging/current_cost���;�nfG+       ��K	Bj]��A�?*

logging/current_cost���;`���+       ��K	�]��A�?*

logging/current_cost}��;�ϫ�+       ��K	��]��A�?*

logging/current_cost���;����+       ��K	X^��A�?*

logging/current_cost׋�;�:3l+       ��K	�?^��A�?*

logging/current_cost���;;ɽK+       ��K	q^��A�?*

logging/current_costd��;~߇�+       ��K	U�^��A�?*

logging/current_cost��;�k,+       ��K	��^��A�?*

logging/current_cost��;�0�b+       ��K	��^��A�?*

logging/current_cost銄;��+�+       ��K	�1_��A�?*

logging/current_cost=��;��U�+       ��K	Hb_��A�?*

logging/current_cost��;�x�O+       ��K	��_��A�?*

logging/current_cost-��;�+       ��K	|�_��A�?*

logging/current_costȍ�;8�k�+       ��K	�`��A�@*

logging/current_cost��;<��+       ��K	>`��A�@*

logging/current_costɉ�;���+       ��K	�o`��A�@*

logging/current_cost���;�� +       ��K	�`��A�@*

logging/current_costI��;.��j+       ��K	��`��A�@*

logging/current_cost���;ZZ+       ��K	��`��A�@*

logging/current_cost싄;�t;2+       ��K	�5a��A�@*

logging/current_cost
��;j�j+       ��K	�ha��A�@*

logging/current_cost茄;�1-+       ��K	�a��A�@*

logging/current_cost劄;g���+       ��K	 �a��A�@*

logging/current_cost���;��P+       ��K	��a��A�@*

logging/current_cost}��;w�_�+       ��K	_/b��A�@*

logging/current_cost،�;�7�+       ��K	�fb��A�@*

logging/current_cost ��;(B>+       ��K	��b��A�@*

logging/current_cost���;�x��+       ��K	��b��A�@*

logging/current_cost(��;���Z+       ��K	f�b��A�@*

logging/current_cost���;�f�+       ��K	C/c��A�@*

logging/current_cost9��;�ܘ%+       ��K	vac��A�@*

logging/current_cost��;7�X�+       ��K	��c��A�@*

logging/current_costF��;�;�+       ��K	��c��A�@*

logging/current_cost	��;� ��+       ��K	c�c��A�@*

logging/current_costy��;��e�+       ��K	20d��A�@*

logging/current_cost���;O��&+       ��K	�md��A�@*

logging/current_cost���;���+       ��K	�d��A�@*

logging/current_cost���;�%}i+       ��K	��d��A�@*

logging/current_cost���;�o�+       ��K	Ne��A�A*

logging/current_costZ��;���^+       ��K	�;e��A�A*

logging/current_cost��;�/o�+       ��K	
ve��A�A*

logging/current_cost*��;�*�+       ��K	��e��A�A*

logging/current_costC��;=��+       ��K	~�e��A�A*

logging/current_costr��;$���+       ��K	�f��A�A*

logging/current_cost���;}?�v+       ��K	�Kf��A�A*

logging/current_costE��;^Qx�+       ��K	��f��A�A*

logging/current_cost莄;O-�+       ��K	\�f��A�A*

logging/current_cost틄;�)��+       ��K	�f��A�A*

logging/current_cost��;GE7+       ��K	�'g��A�A*

logging/current_costǌ�;��7+       ��K	Tgg��A�A*

logging/current_costa��;��+       ��K	��g��A�A*

logging/current_costC��;��Zu+       ��K	��g��A�A*

logging/current_cost���;�-Օ+       ��K	�3h��A�A*

logging/current_cost���; �H�+       ��K	jih��A�A*

logging/current_cost���;�yO�+       ��K	��h��A�A*

logging/current_costA��;�f�M+       ��K	�h��A�A*

logging/current_cost���;u!��+       ��K	(i��A�A*

logging/current_costL��;���+       ��K	�hi��A�A*

logging/current_costD��;��+       ��K	��i��A�A*

logging/current_cost~��;���I+       ��K	�j��A�A*

logging/current_costD��;}���+       ��K	�Gj��A�A*

logging/current_cost���;ޠ
e+       ��K	E�j��A�A*

logging/current_cost6��;����+       ��K	�j��A�A*

logging/current_costf��;��2+       ��K	�k��A�A*

logging/current_costV��;2 �M+       ��K	Ik��A�B*

logging/current_cost��;�`G�+       ��K	��k��A�B*

logging/current_costy��;~��+       ��K	��k��A�B*

logging/current_cost��;ʚ��+       ��K	.	l��A�B*

logging/current_cost��;G�~+       ��K	MHl��A�B*

logging/current_costV��;�N��+       ��K	��l��A�B*

logging/current_costs��;�@V�+       ��K	��l��A�B*

logging/current_cost���;y֎�+       ��K	L�l��A�B*

logging/current_cost䋄;��+       ��K	�1m��A�B*

logging/current_cost���;���+       ��K	�wm��A�B*

logging/current_cost���;n5t�+       ��K	4�m��A�B*

logging/current_cost���;X��v+       ��K	 �m��A�B*

logging/current_cost�;L��+       ��K	�&n��A�B*

logging/current_cost���;�ɓo+       ��K	 [n��A�B*

logging/current_cost���;6��+       ��K	��n��A�B*

logging/current_cost���;?��+       ��K	Թn��A�B*

logging/current_costމ�;�Z+       ��K	e�n��A�B*

logging/current_cost��;p��+       ��K	K o��A�B*

logging/current_cost��;^2~+       ��K	�[o��A�B*

logging/current_costW��;3Ť+       ��K	Ԋo��A�B*

logging/current_cost@��;-?��+       ��K	Ѿo��A�B*

logging/current_cost��;RZ�P+       ��K	��o��A�B*

logging/current_cost���;_а�+       ��K	�"p��A�B*

logging/current_cost��;�_{+       ��K	�Qp��A�B*

logging/current_cost���;R��&+       ��K	3�p��A�B*

logging/current_cost�;[i46+       ��K	�p��A�B*

logging/current_costH��;᫫W+       ��K	��p��A�C*

logging/current_cost|��;+u$L+       ��K	n"q��A�C*

logging/current_costW��;DE:�+       ��K	KVq��A�C*

logging/current_cost;��;،]+       ��K	�q��A�C*

logging/current_cost���;@��+       ��K	1�q��A�C*

logging/current_costߎ�;ݕ��+       ��K	~�q��A�C*

logging/current_cost��;�} �+       ��K	�\r��A�C*

logging/current_cost=��;GӋ+       ��K	m�r��A�C*

logging/current_costj��;zK�+       ��K	�r��A�C*

logging/current_cost�;�Y�+       ��K	��r��A�C*

logging/current_cost���;���6+       ��K	O9s��A�C*

logging/current_costC��;�PǙ+       ��K	�zs��A�C*

logging/current_cost���;�� �+       ��K	ݯs��A�C*

logging/current_cost��;�-j�+       ��K	��s��A�C*

logging/current_cost!��;���t+       ��K	�+t��A�C*

logging/current_cost���;�ϴ+       ��K	�^t��A�C*

logging/current_cost���;{�I#+       ��K	*�t��A�C*

logging/current_cost��;y���+       ��K	q�t��A�C*

logging/current_costn��;�I#�+       ��K	�u��A�C*

logging/current_cost0��;3�$+       ��K	�<u��A�C*

logging/current_cost;�rub+       ��K	yvu��A�C*

logging/current_cost���;8��9+       ��K	��u��A�C*

logging/current_costE��;�0�"+       ��K	I�u��A�C*

logging/current_costǉ�;%���+       ��K	 v��A�C*

logging/current_cost���;�)�O+       ��K	�:v��A�C*

logging/current_cost ��;��@r+       ��K	�kv��A�D*

logging/current_cost_��;���5+       ��K	x�v��A�D*

logging/current_cost���;���;+       ��K	�v��A�D*

logging/current_cost���;���^+       ��K	4�v��A�D*

logging/current_cost���;[�s�+       ��K	�+w��A�D*

logging/current_cost͌�;Qp+       ��K	P\w��A�D*

logging/current_cost���;���#+       ��K	R�w��A�D*

logging/current_cost���;<=j�+       ��K	&�w��A�D*

logging/current_cost���;FO��+       ��K	�w��A�D*

logging/current_costf��;��u�+       ��K	�x��A�D*

logging/current_cost���;���i+       ��K	�Px��A�D*

logging/current_costp��;;�T�+       ��K	F�x��A�D*

logging/current_cost���;���+       ��K	0�x��A�D*

logging/current_cost��;��1+       ��K	)�x��A�D*

logging/current_cost̌�;�&]+       ��K	�y��A�D*

logging/current_cost���;\M�X+       ��K	Cy��A�D*

logging/current_cost��;�Ο+       ��K	�sy��A�D*

logging/current_costY��; �p>+       ��K	ϣy��A�D*

logging/current_cost���; �4+       ��K	��y��A�D*

logging/current_cost׊�;'R^+       ��K	dz��A�D*

logging/current_costЉ�;�-2�+       ��K	f1z��A�D*

logging/current_cost���;�&�{+       ��K	�]z��A�D*

logging/current_cost���;���+       ��K	֌z��A�D*

logging/current_cost���;���+       ��K	ͻz��A�D*

logging/current_cost���;z�d�+       ��K	��z��A�D*

logging/current_cost늄;��" +       ��K	�{��A�D*

logging/current_costL��;�C+       ��K	XK{��A�E*

logging/current_cost]��;�'l�+       ��K	��{��A�E*

logging/current_cost(��;S�I�+       ��K	��{��A�E*

logging/current_cost���;8�M+       ��K	�/|��A�E*

logging/current_cost���;&��-+       ��K	�l|��A�E*

logging/current_costs��;�*�I+       ��K	V�|��A�E*

logging/current_costȊ�;�)O+       ��K	��|��A�E*

logging/current_cost��;�+�+       ��K	?.}��A�E*

logging/current_costj��;�\��+       ��K	+r}��A�E*

logging/current_cost%��;t�A+       ��K	:�}��A�E*

logging/current_cost(��;�ƾ+       ��K	,~��A�E*

logging/current_cost;�sg+       ��K	)t~��A�E*

logging/current_costW��;��P+       ��K	u�~��A�E*

logging/current_cost�;�� D+       ��K	��~��A�E*

logging/current_cost荄;�}4+       ��K	{��A�E*

logging/current_cost.��;97Q�+       ��K	�M��A�E*

logging/current_cost��;���"+       ��K	���A�E*

logging/current_cost���;"��1+       ��K	���A�E*

logging/current_cost���;��9D+       ��K	H���A�E*

logging/current_cost���;�$��+       ��K	����A�E*

logging/current_cost=��;]�D+       ��K	S{���A�E*

logging/current_costč�;Hqr+       ��K	ְ���A�E*

logging/current_cost���;<w��+       ��K	�怇�A�E*

logging/current_costۊ�;n���+       ��K	����A�E*

logging/current_costދ�;xL<�+       ��K	W���A�E*

logging/current_cost~��;m�+       ��K	�����A�F*

logging/current_cost0��;��L+       ��K	΁��A�F*

logging/current_cost��;b�y+       ��K	
���A�F*

logging/current_costɌ�;�5�+       ��K	�G���A�F*

logging/current_cost���;+y-+       ��K	á���A�F*

logging/current_costO��;���f+       ��K	;Ԃ��A�F*

logging/current_cost���;��+       ��K	����A�F*

logging/current_cost	��;p�K�+       ��K	�E���A�F*

logging/current_costۋ�;�D�+       ��K	���A�F*

logging/current_cost���;�y8w+       ��K	����A�F*

logging/current_cost퉄;Z�g�+       ��K	�僇�A�F*

logging/current_cost��;!��c+       ��K	�%���A�F*

logging/current_cost{��;4{ZS+       ��K	=b���A�F*

logging/current_costZ��;�A�+       ��K	����A�F*

logging/current_costꎄ;+�+       ��K	eĄ��A�F*

logging/current_cost��;��	+       ��K	�����A�F*

logging/current_costy��;7�)+       ��K	�&���A�F*

logging/current_cost���;��~�+       ��K	Y���A�F*

logging/current_costU��;���+       ��K	����A�F*

logging/current_cost���;E�#++       ��K	@����A�F*

logging/current_costw��;ޱE+       ��K	�兇�A�F*

logging/current_costÉ�;���+       ��K	U���A�F*

logging/current_costN��;>C�?+       ��K	fB���A�F*

logging/current_costM��;���+       ��K	qw���A�F*

logging/current_cost���;�.�S+       ��K	�����A�F*

logging/current_cost���;`��t+       ��K	UՆ��A�F*

logging/current_costɋ�;$Ȳt+       ��K	���A�G*

logging/current_cost2��;��R.+       ��K	�4���A�G*

logging/current_costN��;y�|�+       ��K	�a���A�G*

logging/current_costc��;�q^�+       ��K	攇��A�G*

logging/current_cost���;���+       ��K	���A�G*

logging/current_costD��;�Q�Y+       ��K	+��A�G*

logging/current_cost���;CT�e+       ��K	W ���A�G*

logging/current_cost��;�\�L+       ��K	M���A�G*

logging/current_cost��;	/�+       ��K	�z���A�G*

logging/current_costn��;<�+       ��K	�����A�G*

logging/current_cost���;s���+       ��K	zሇ�A�G*

logging/current_cost���;�
�+       ��K	"���A�G*

logging/current_cost鋄;� �c+       ��K	�;���A�G*

logging/current_costt��;uZ��+       ��K	�k���A�G*

logging/current_cost叄;7#<+       ��K	����A�G*

logging/current_costy��;��^�+       ��K	�̉��A�G*

logging/current_cost%��;z7P�+       ��K	����A�G*

logging/current_cost���;�R+       ��K	(���A�G*

logging/current_cost,��;(� +       ��K	YT���A�G*

logging/current_costV��;"� +       ��K	����A�G*

logging/current_costъ�;����+       ��K	�����A�G*

logging/current_cost���;���+       ��K	n芇�A�G*

logging/current_cost���;z5�@+       ��K	U���A�G*

logging/current_costt��;H�DM+       ��K	�F���A�G*

logging/current_cost劄;���%+       ��K	�t���A�G*

logging/current_cost:��;8�Wr+       ��K	����A�G*

logging/current_cost���;̚�+       ��K	�Ћ��A�H*

logging/current_cost���;��+       ��K	 ����A�H*

logging/current_cost;��;*�?F+       ��K	),���A�H*

logging/current_costÍ�;ܸq�+       ��K	wY���A�H*

logging/current_costs��;�x<y+       ��K	b����A�H*

logging/current_cost;���+       ��K		����A�H*

logging/current_cost��;JW&�+       ��K	0ꌇ�A�H*

logging/current_cost*��;���X+       ��K	:���A�H*

logging/current_cost��;{�7%+       ��K	&K���A�H*

logging/current_costb��;�!��+       ��K	=y���A�H*

logging/current_cost]��;u%ci+       ��K	�����A�H*

logging/current_costE��;��'�+       ��K	�Ս��A�H*

logging/current_cost<��;����+       ��K	����A�H*

logging/current_cost;.oM�+       ��K	�3���A�H*

logging/current_cost!��;��)�+       ��K	Jc���A�H*

logging/current_costr��;�^��+       ��K	����A�H*

logging/current_costF��;F�+       ��K	P����A�H*

logging/current_costS��;�YP�+       ��K	�뎇�A�H*

logging/current_cost���;:�g+       ��K	����A�H*

logging/current_cost芄;c�C+       ��K	�I���A�H*

logging/current_cost���;`9l�+       ��K	�y���A�H*

logging/current_costt��;8!�~+       ��K	ۨ���A�H*

logging/current_cost���;y��9+       ��K	H؏��A�H*

logging/current_costY��;m�Ғ+       ��K		I���A�H*

logging/current_cost\��;�8�+       ��K	悐��A�H*

logging/current_cost͋�;�U
m+       ��K	乐��A�I*

logging/current_cost͋�;��E�+       ��K	�����A�I*

logging/current_cost���;/�+       ��K	�2���A�I*

logging/current_cost���;p�ʋ+       ��K	s���A�I*

logging/current_cost���;���+       ��K	«���A�I*

logging/current_cost���; �44+       ��K	r쑇�A�I*

logging/current_cost��;�?��+       ��K	�&���A�I*

logging/current_costp��;sHs+       ��K	�\���A�I*

logging/current_cost댄;���}+       ��K	ؐ���A�I*

logging/current_costP��;���+       ��K	jŒ��A�I*

logging/current_cost늄;k�k�+       ��K	5����A�I*

logging/current_cost7��;�m�+       ��K	n6���A�I*

logging/current_costy��;ָ�K+       ��K	Ni���A�I*

logging/current_cost퉄;A��c+       ��K	�����A�I*

logging/current_cost���; �g�+       ��K	f̓��A�I*

logging/current_costމ�;1��+       ��K	7����A�I*

logging/current_cost���;�UoQ+       ��K	b)���A�I*

logging/current_cost���;?(7+       ��K	>X���A�I*

logging/current_cost��;�pƢ+       ��K	;����A�I*

logging/current_costS��;3.��+       ��K	�����A�I*

logging/current_costB��;�?+       ��K	�攇�A�I*

logging/current_cost?��;F�ؙ+       ��K	`���A�I*

logging/current_cost���;6:�+       ��K	�B���A�I*

logging/current_cost_��;�/ W+       ��K	�x���A�I*

logging/current_cost#��;��+       ��K	����A�I*

logging/current_cost��;�I�0+       ��K	�ؕ��A�I*

logging/current_cost卄;1��+       ��K	�	���A�J*

logging/current_cost㋄;�3�+       ��K	o6���A�J*

logging/current_cost���;��+       ��K	ad���A�J*

logging/current_costҋ�;�^�+       ��K	�����A�J*

logging/current_cost���;4�,K+       ��K	Z֖��A�J*

logging/current_cost`��;I���+       ��K	Q���A�J*

logging/current_cost���;$`eA+       ��K	f3���A�J*

logging/current_cost���;W `C+       ��K	'`���A�J*

logging/current_cost��;_��2+       ��K	 ����A�J*

logging/current_cost׌�;~�/'+       ��K	<ė��A�J*

logging/current_cost��;����+       ��K	�����A�J*

logging/current_cost��;؎D�+       ��K	�$���A�J*

logging/current_cost,��;�� �+       ��K	�R���A�J*

logging/current_cost1��;_�u�+       ��K	ʀ���A�J*

logging/current_cost���;���+       ��K	2����A�J*

logging/current_costŊ�;)��+       ��K	;阇�A�J*

logging/current_cost��;�H=�+       ��K	����A�J*

logging/current_costT��;�]�+       ��K	WC���A�J*

logging/current_costy��;'".+       ��K	�t���A�J*

logging/current_cost�;��+       ��K	�����A�J*

logging/current_cost_��;	��+       ��K	)ә��A�J*

logging/current_cost��;k;Y+       ��K	d���A�J*

logging/current_costꎄ;\�h+       ��K	t6���A�J*

logging/current_costS��;�R��+       ��K	;d���A�J*

logging/current_cost���;BZ��+       ��K	ԑ���A�J*

logging/current_costg��;-Y�+       ��K	�Ě��A�K*

logging/current_cost	��;k�E�+       ��K	2����A�K*

logging/current_costʌ�;[��+       ��K	 %���A�K*

logging/current_costꋄ;�&�u+       ��K	fT���A�K*

logging/current_cost;��;v_r\+       ��K	m����A�K*

logging/current_cost���;%A�+       ��K	�����A�K*

logging/current_costH��;���F+       ��K	3훇�A�K*

logging/current_cost��;X�.+       ��K	����A�K*

logging/current_cost���;���{+       ��K	tL���A�K*

logging/current_cost4��;OY=+       ��K	�y���A�K*

logging/current_costo��;��Rr+       ��K	2����A�K*

logging/current_cost���;Ӻ�+       ��K	~ڜ��A�K*

logging/current_cost"��;��\�+       ��K	R+���A�K*

logging/current_cost���;�:�+       ��K	Y���A�K*

logging/current_cost鋄;#о�+       ��K	̅���A�K*

logging/current_costގ�;宖+       ��K	�����A�K*

logging/current_cost���;=?��+       ��K	s均�A�K*

logging/current_cost/��;>��+       ��K	U���A�K*

logging/current_cost���;L�$M+       ��K	�F���A�K*

logging/current_costΌ�;C&��+       ��K	�t���A�K*

logging/current_cost���;nT�+       ��K	<����A�K*

logging/current_costq��;�+       ��K	�О��A�K*

logging/current_cost1��;�+��+       ��K	����A�K*

logging/current_cost≄;�+       ��K	�/���A�K*

logging/current_cost���;zj�C+       ��K	_���A�K*

logging/current_cost���;r�7'+       ��K	T����A�K*

logging/current_costv��;�@��+       ��K	�����A�L*

logging/current_cost���;S+       ��K	쟇�A�L*

logging/current_costl��;=T�4+       ��K	$!���A�L*

logging/current_cost|��;YБn+       ��K	bN���A�L*

logging/current_costC��;��o�+       ��K	�{���A�L*

logging/current_cost���;�m+       ��K	稠��A�L*

logging/current_costp��;2�a4+       ��K	Gؠ��A�L*

logging/current_cost슄;,�c+       ��K	����A�L*

logging/current_cost���;�x�+       ��K	�3���A�L*

logging/current_cost'��;D	�+       ��K	�a���A�L*

logging/current_cost��;���+       ��K	ӏ���A�L*

logging/current_cost���;��ό+       ��K	{����A�L*

logging/current_cost���;Ȍdh+       ��K	N졇�A�L*

logging/current_cost���;m+hq+       ��K	����A�L*

logging/current_costǍ�;mF�+       ��K	�F���A�L*

logging/current_cost���;,�+       ��K	Zu���A�L*

logging/current_cost���;`º6+       ��K	E����A�L*

logging/current_cost���;�P�*+       ��K	�Ԣ��A�L*

logging/current_cost���;Sb��+       ��K	���A�L*

logging/current_cost���;r'^+       ��K	d0���A�L*

logging/current_cost׉�;єOb+       ��K	�]���A�L*

logging/current_coste��;�rJz+       ��K	M����A�L*

logging/current_costE��;дև+       ��K	x����A�L*

logging/current_cost���;��+       ��K	\䣇�A�L*

logging/current_cost;�A��+       ��K	M���A�L*

logging/current_cost���;w"��+       ��K	`?���A�L*

logging/current_cost ��;k8{g+       ��K	�l���A�M*

logging/current_cost&��;^j2(+       ��K	w����A�M*

logging/current_cost���;���7+       ��K	�̤��A�M*

logging/current_cost͊�;����+       ��K	Q����A�M*

logging/current_cost��;����+       ��K	�-���A�M*

logging/current_cost*��;m��+       ��K	�[���A�M*

logging/current_costō�;Pl�U+       ��K	F����A�M*

logging/current_cost.��;G�g+       ��K	����A�M*

logging/current_costˋ�; ��W+       ��K	�쥇�A�M*

logging/current_cost���;]r|�+       ��K	*���A�M*

logging/current_cost֋�;�k��+       ��K	�H���A�M*

logging/current_cost���;m��k+       ��K	ty���A�M*

logging/current_costʋ�;�Q�+       ��K	�����A�M*

logging/current_cost��;^~]�+       ��K	�ڦ��A�M*

logging/current_costǋ�;�4�8+       ��K	A���A�M*

logging/current_costU��;����+       ��K	;���A�M*

logging/current_cost��;�;��+       ��K	$j���A�M*

logging/current_cost1��;jh%�+       ��K	/����A�M*

logging/current_cost���;��`+       ��K	[ɧ��A�M*

logging/current_cost��;/q++       ��K	F����A�M*

logging/current_cost���;4u7�+       ��K	�%���A�M*

logging/current_costX��;)��+       ��K	T���A�M*

logging/current_cost��;�[�+       ��K	�����A�M*

logging/current_cost���;YXՏ+       ��K	����A�M*

logging/current_cost��;F���+       ��K	�ਇ�A�M*

logging/current_costE��;���+       ��K	����A�N*

logging/current_cost���;��mU+       ��K	BB���A�N*

logging/current_cost싄;h��o+       ��K	1s���A�N*

logging/current_cost7��;׿�@